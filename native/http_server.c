// Minimal epoll-based HTTP/1.1 server for the fraud-detection backend.
//
// Runs entirely in C — Bun spawns this via FFI and then idles. No Bun.serve
// wrapper, no Request/Response allocation per request. Hot path:
//   epoll → accept → read → handle_http() → write → loop
//
// Connection state is a small struct per fd. Connections are stored in a
// flat array indexed by a hash of fd (or scanned linearly — for the
// contest's max ~250 VUs this is fine).
//
// Listens on a Unix domain socket. The `start_http_server` entry point
// spawns a pthread and returns; the caller (Bun) just stays alive.
// Linux-only (epoll, accept4) — defined before any system header.
#define _GNU_SOURCE 1

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/epoll.h>
#include <sys/stat.h>

// External: defined in process.c
extern int32_t handle_http(
    const char *req, int32_t req_len,
    char *resp, int32_t resp_cap,
    float *query_buf
);
extern void set_ready(int32_t r);

#define MAX_CONNS 1024
#define READ_BUF_INITIAL  4096
#define READ_BUF_MAX      32768
#define WRITE_BUF_SIZE    4096
#define EPOLL_BATCH       64

typedef struct {
    int32_t   fd;          // -1 if free slot
    char     *read_buf;
    int32_t   read_cap;
    int32_t   read_len;    // bytes currently in buffer
    char     *write_buf;   // pending write (when write would block)
    int32_t   write_len;   // total bytes to write
    int32_t   write_off;   // bytes already written
    float    *query_buf;   // per-conn scratch (pinned at init)
} conn_t;

typedef struct {
    int                listen_fd;
    int                epoll_fd;
    conn_t            *conns;          // [MAX_CONNS]
    pthread_t          thread;
    char               sock_path[108];
} srv_t;

static srv_t G_SRV;

// --- Helpers --------------------------------------------------------------


static conn_t *acquire_conn(int fd) {
    for (int i = 0; i < MAX_CONNS; i++) {
        if (G_SRV.conns[i].fd < 0) {
            conn_t *c = &G_SRV.conns[i];
            c->fd = fd;
            c->read_len = 0;
            c->write_len = 0;
            c->write_off = 0;
            if (!c->read_buf) {
                c->read_buf = (char *)malloc(READ_BUF_INITIAL);
                c->read_cap = READ_BUF_INITIAL;
            }
            if (!c->write_buf) {
                c->write_buf = (char *)malloc(WRITE_BUF_SIZE);
            }
            if (!c->query_buf) {
                // 16 floats (D=14 + 2 padding) — matches process_request expectation
                c->query_buf = (float *)aligned_alloc(64, sizeof(float) * 16);
                memset(c->query_buf, 0, sizeof(float) * 16);
            }
            return c;
        }
    }
    return NULL;
}

static void release_conn(conn_t *c) {
    if (c->fd >= 0) {
        epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_DEL, c->fd, NULL);
        close(c->fd);
        c->fd = -1;
    }
    c->read_len = 0;
    c->write_len = 0;
    c->write_off = 0;
}

static conn_t *find_conn(int fd) {
    for (int i = 0; i < MAX_CONNS; i++) {
        if (G_SRV.conns[i].fd == fd) return &G_SRV.conns[i];
    }
    return NULL;
}

// Try to flush pending write. Returns 0 on success (all written), 1 on
// would-block (re-arm EPOLLOUT), -1 on error (close conn).
static int try_flush_write(conn_t *c) {
    while (c->write_off < c->write_len) {
        ssize_t n = write(c->fd, c->write_buf + c->write_off, c->write_len - c->write_off);
        if (n > 0) {
            c->write_off += (int32_t)n;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 1;
        } else {
            return -1;
        }
    }
    c->write_len = 0;
    c->write_off = 0;
    return 0;
}

// Try to drain as many full HTTP requests from read_buf as possible.
// Returns 0 on success, 1 on would-block (re-arm EPOLLOUT), -1 on error.
static int drain_requests(conn_t *c) {
    while (c->read_len > 0) {
        // If we still have pending writes, finish those first.
        if (c->write_len > c->write_off) {
            int r = try_flush_write(c);
            if (r != 0) return r;
        }

        int32_t written = handle_http(
            c->read_buf, c->read_len,
            c->write_buf, WRITE_BUF_SIZE,
            c->query_buf
        );
        if (written == 0) return 0;       // need more bytes
        if (written < 0)  return -1;      // protocol error → close

        // Determine how many bytes the request consumed.
        // Mirror logic in handle_http: find \r\n\r\n + content-length.
        int body_start = -1;
        for (int i = 0; i + 3 < c->read_len; i++) {
            if (c->read_buf[i] == '\r' && c->read_buf[i+1] == '\n' &&
                c->read_buf[i+2] == '\r' && c->read_buf[i+3] == '\n') {
                body_start = i + 4;
                break;
            }
        }
        if (body_start < 0) return 0;

        int consumed;
        if (c->read_buf[0] == 'G') {
            // GET — no body
            consumed = body_start;
        } else {
            // POST — find Content-Length
            int cl = -1;
            for (int i = 0; i + 17 < body_start; i++) {
                const char *p = c->read_buf + i;
                if (p[0] == '\r' && p[1] == '\n' &&
                    (memcmp(p+2, "Content-Length:", 15) == 0 ||
                     memcmp(p+2, "content-length:", 15) == 0)) {
                    int j = i + 17;
                    while (j < body_start && c->read_buf[j] == ' ') j++;
                    int v = 0;
                    while (j < body_start && c->read_buf[j] >= '0' && c->read_buf[j] <= '9') {
                        v = v * 10 + (c->read_buf[j] - '0');
                        j++;
                    }
                    cl = v;
                    break;
                }
            }
            if (cl < 0) return -1;
            consumed = body_start + cl;
        }

        if (c->read_len < consumed) return 0; // body not all in yet

        // Write response.
        c->write_len = written;
        c->write_off = 0;
        int r = try_flush_write(c);
        if (r < 0) return -1;
        // r == 1 means partial write; we'll come back via EPOLLOUT.

        // Shift remaining bytes to front of read_buf.
        if (consumed == c->read_len) {
            c->read_len = 0;
        } else {
            memmove(c->read_buf, c->read_buf + consumed, c->read_len - consumed);
            c->read_len -= consumed;
        }

        if (r == 1) return 1; // need EPOLLOUT before next request
    }
    return 0;
}

// Read available bytes into c->read_buf. Returns 0 on success, -1 on close
// (peer EOF or error), 1 on would-block (re-arm EPOLLIN).
static int try_read(conn_t *c) {
    for (;;) {
        // Grow buffer if needed.
        if (c->read_len == c->read_cap) {
            if (c->read_cap >= READ_BUF_MAX) return -1; // request too big
            int32_t new_cap = c->read_cap * 2;
            char *nb = (char *)realloc(c->read_buf, new_cap);
            if (!nb) return -1;
            c->read_buf = nb;
            c->read_cap = new_cap;
        }
        ssize_t n = read(c->fd, c->read_buf + c->read_len, c->read_cap - c->read_len);
        if (n > 0) {
            c->read_len += (int32_t)n;
        } else if (n == 0) {
            return -1; // peer closed
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 1;
        } else {
            return -1;
        }
    }
}

// --- Main loop ------------------------------------------------------------

static void *server_thread(void *arg) {
    (void)arg;
    struct epoll_event ev;
    struct epoll_event events[EPOLL_BATCH];

    // Register listener.
    ev.events = EPOLLIN;
    ev.data.fd = G_SRV.listen_fd;
    if (epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_ADD, G_SRV.listen_fd, &ev) < 0) {
        fprintf(stderr, "[http_server] epoll_ctl ADD listener failed: %s\n", strerror(errno));
        return NULL;
    }

    fprintf(stderr, "[http_server] event loop started\n");

    for (;;) {
        int n = epoll_wait(G_SRV.epoll_fd, events, EPOLL_BATCH, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "[http_server] epoll_wait: %s\n", strerror(errno));
            break;
        }
        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;
            uint32_t evbits = events[i].events;

            if (fd == G_SRV.listen_fd) {
                // Accept all pending connections.
                for (;;) {
                    int cfd = accept4(G_SRV.listen_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
                    if (cfd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        fprintf(stderr, "[http_server] accept4: %s\n", strerror(errno));
                        break;
                    }
                    conn_t *c = acquire_conn(cfd);
                    if (!c) {
                        close(cfd);
                        continue;
                    }
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = cfd;
                    if (epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_ADD, cfd, &ev) < 0) {
                        release_conn(c);
                    }
                }
                continue;
            }

            conn_t *c = find_conn(fd);
            if (!c) {
                close(fd);
                continue;
            }

            if (evbits & (EPOLLERR | EPOLLHUP)) {
                release_conn(c);
                continue;
            }

            if (evbits & EPOLLIN) {
                int rr = try_read(c);
                if (rr < 0) {
                    release_conn(c);
                    continue;
                }
                int dr = drain_requests(c);
                if (dr < 0) {
                    release_conn(c);
                    continue;
                }
                if (dr == 1) {
                    // Pending write; arm EPOLLOUT.
                    ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
                    ev.data.fd = fd;
                    epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_MOD, fd, &ev);
                }
            }

            if (evbits & EPOLLOUT) {
                int wr = try_flush_write(c);
                if (wr < 0) {
                    release_conn(c);
                    continue;
                }
                // After flushing, try to drain any further requests.
                int dr = drain_requests(c);
                if (dr < 0) {
                    release_conn(c);
                    continue;
                }
                if (dr == 0) {
                    // No more pending output → drop EPOLLOUT.
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = fd;
                    epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_MOD, fd, &ev);
                }
            }
        }
    }
    return NULL;
}

// --- Public entry ---------------------------------------------------------

// Bind a listening UDS at sock_path, register with epoll, spawn the event
// loop thread, and return immediately. Returns 0 on success.
int32_t start_http_server(const char *sock_path) {
    memset(&G_SRV, 0, sizeof(G_SRV));

    // Allocate connection table.
    G_SRV.conns = (conn_t *)calloc(MAX_CONNS, sizeof(conn_t));
    if (!G_SRV.conns) return -1;
    for (int i = 0; i < MAX_CONNS; i++) G_SRV.conns[i].fd = -1;

    // Stale socket cleanup.
    unlink(sock_path);

    // Bind.
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        fprintf(stderr, "[http_server] socket: %s\n", strerror(errno));
        return -1;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[http_server] bind %s: %s\n", sock_path, strerror(errno));
        close(fd);
        return -1;
    }
    chmod(sock_path, 0666);
    if (listen(fd, 1024) < 0) {
        fprintf(stderr, "[http_server] listen: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    G_SRV.listen_fd = fd;
    strncpy(G_SRV.sock_path, sock_path, sizeof(G_SRV.sock_path) - 1);

    // epoll.
    G_SRV.epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (G_SRV.epoll_fd < 0) {
        fprintf(stderr, "[http_server] epoll_create1: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    // Spawn thread.
    if (pthread_create(&G_SRV.thread, NULL, server_thread, NULL) != 0) {
        fprintf(stderr, "[http_server] pthread_create failed\n");
        close(G_SRV.epoll_fd);
        close(fd);
        return -1;
    }

    fprintf(stderr, "[http_server] listening on %s\n", sock_path);
    return 0;
}
