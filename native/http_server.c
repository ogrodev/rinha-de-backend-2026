// Minimal epoll-based HTTP/1.1 server for the fraud-detection backend.
//
// Runs entirely in C — Bun spawns this via FFI and then idles. No Bun.serve
// wrapper, no Request/Response allocation per request. Hot path:
//   epoll → accept → read → handle_http() → write → loop
//
// Connection state is dynamically allocated per accept and stored directly
// on epoll_event.data.ptr — O(1) lookup, no scan. Buffers are pre-sized to
// 4 KB (large enough for the contest payload) and reused across requests on
// the same keep-alive connection.

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

extern int32_t handle_http(
    const char *req, int32_t req_len,
    char *resp, int32_t resp_cap,
    float *query_buf
);
extern int32_t handle_http_v2(
    const char *req, int32_t req_len,
    char *resp, int32_t resp_cap,
    float *query_buf,
    int32_t *consumed_out
);

#define READ_BUF_SIZE   4096
#define WRITE_BUF_SIZE  4096
#define EPOLL_BATCH     128

typedef struct {
    int       fd;
    char     *read_buf;
    int32_t   read_len;
    char     *write_buf;
    int32_t   write_len;
    int32_t   write_off;
    float    *query_buf;
} conn_t;

typedef struct {
    int                listen_fd;
    int                epoll_fd;
    pthread_t          thread;
} srv_t;
static srv_t G_SRV;

// --- Connection pool --------------------------------------------------------
//
// Pre-allocate up to MAX_POOL conn_t structs (with their buffers) at start.
// acquire/release use a simple stack of free indices. Eliminates malloc/free
// from the per-request hot path; under contest load (~250 max VUs) we
// rarely exceed 256 simultaneous connections.

#define MAX_POOL 512

static conn_t G_POOL[MAX_POOL];
static int    G_FREE[MAX_POOL];
static int    G_FREE_TOP = 0;
static int    G_POOL_INIT = 0;

static void pool_init(void) {
    for (int i = 0; i < MAX_POOL; i++) {
        G_POOL[i].fd = -1;
        G_POOL[i].read_buf = (char *)malloc(READ_BUF_SIZE);
        G_POOL[i].write_buf = (char *)malloc(WRITE_BUF_SIZE);
        G_POOL[i].query_buf = (float *)aligned_alloc(64, sizeof(float) * 16);
        memset(G_POOL[i].query_buf, 0, sizeof(float) * 16);
        G_FREE[i] = i;
    }
    G_FREE_TOP = MAX_POOL;
    G_POOL_INIT = 1;
}

static conn_t *pool_acquire(int fd) {
    if (G_FREE_TOP == 0) return NULL;
    int idx = G_FREE[--G_FREE_TOP];
    conn_t *c = &G_POOL[idx];
    c->fd = fd;
    c->read_len = 0;
    c->write_len = 0;
    c->write_off = 0;
    return c;
}

static void pool_release(conn_t *c) {
    int idx = (int)(c - G_POOL);
    if (idx < 0 || idx >= MAX_POOL) return;
    c->fd = -1;
    G_FREE[G_FREE_TOP++] = idx;
}

// --- Per-connection lifecycle -----------------------------------------------

static conn_t *conn_new(int fd) {
    return pool_acquire(fd);
}

static void conn_close(conn_t *c) {
    if (!c) return;
    if (c->fd >= 0) {
        epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_DEL, c->fd, NULL);
        close(c->fd);
    }
    pool_release(c);
}

// --- I/O helpers ------------------------------------------------------------

// 0 = all written, 1 = would block (re-arm EPOLLOUT), -1 = error.
static int try_flush_write(conn_t *c) {
    while (c->write_off < c->write_len) {
        ssize_t n = write(c->fd, c->write_buf + c->write_off,
                          c->write_len - c->write_off);
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

// 0 = need more bytes / nothing more to do, 1 = pending write blocked, -1 = error.
static int drain_requests(conn_t *c) {
    while (c->read_len > 0) {
        if (c->write_len > c->write_off) {
            int r = try_flush_write(c);
            if (r != 0) return r;
        }

        int32_t consumed = 0;
        int32_t written = handle_http_v2(
            c->read_buf, c->read_len,
            c->write_buf, WRITE_BUF_SIZE,
            c->query_buf,
            &consumed
        );
        if (written == 0) return 0;
        if (written < 0)  return -1;
        if (consumed <= 0) return -1;

        c->write_len = written;
        c->write_off = 0;
        int r = try_flush_write(c);
        if (r < 0) return -1;

        if (consumed == c->read_len) {
            c->read_len = 0;
        } else {
            memmove(c->read_buf, c->read_buf + consumed, c->read_len - consumed);
            c->read_len -= consumed;
        }
        if (r == 1) return 1;
    }
    return 0;
}

// 0 = read some, 1 = would block, -1 = peer closed / error.
static int try_read(conn_t *c) {
    for (;;) {
        if (c->read_len >= READ_BUF_SIZE) {
            // Request larger than our buffer.
            return -1;
        }
        ssize_t n = read(c->fd, c->read_buf + c->read_len,
                         READ_BUF_SIZE - c->read_len);
        if (n > 0) {
            c->read_len += (int32_t)n;
        } else if (n == 0) {
            return -1;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 1;
        } else {
            return -1;
        }
    }
}

// --- Main loop --------------------------------------------------------------

static void *server_thread(void *arg) {
    (void)arg;
    struct epoll_event ev;
    struct epoll_event events[EPOLL_BATCH];


    ev.events = EPOLLIN;
    ev.data.ptr = NULL; // listener: special-cased by ptr==NULL
    if (epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_ADD, G_SRV.listen_fd, &ev) < 0) {
        fprintf(stderr, "[http_server] epoll_ctl ADD listener: %s\n", strerror(errno));
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
            uint32_t evbits = events[i].events;
            conn_t *c = (conn_t *)events[i].data.ptr;

            if (c == NULL) {
                // Listener — accept all pending.
                for (;;) {
                    int cfd = accept4(G_SRV.listen_fd, NULL, NULL,
                                      SOCK_NONBLOCK | SOCK_CLOEXEC);
                    if (cfd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        fprintf(stderr, "[http_server] accept4: %s\n", strerror(errno));
                        break;
                    }
                    conn_t *nc = conn_new(cfd);
                    if (!nc) {
                        close(cfd);
                        continue;
                    }
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.ptr = nc;
                    if (epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_ADD, cfd, &ev) < 0) {
                        conn_close(nc);
                    }
                }
                continue;
            }

            if (evbits & (EPOLLERR | EPOLLHUP)) {
                conn_close(c);
                continue;
            }

            if (evbits & EPOLLIN) {
                int rr = try_read(c);
                if (rr < 0) {
                    conn_close(c);
                    continue;
                }
                int dr = drain_requests(c);
                if (dr < 0) {
                    conn_close(c);
                    continue;
                }
                if (dr == 1) {
                    ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
                    ev.data.ptr = c;
                    epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_MOD, c->fd, &ev);
                }
            }

            if (evbits & EPOLLOUT) {
                int wr = try_flush_write(c);
                if (wr < 0) {
                    conn_close(c);
                    continue;
                }
                int dr = drain_requests(c);
                if (dr < 0) {
                    conn_close(c);
                    continue;
                }
                if (dr == 0) {
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.ptr = c;
                    epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_MOD, c->fd, &ev);
                }
            }
        }
    }
    return NULL;
}

// --- Public entry -----------------------------------------------------------

static int setup_listener(const char *sock_path) {
    memset(&G_SRV, 0, sizeof(G_SRV));
    if (!G_POOL_INIT) pool_init();

    unlink(sock_path);

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
    if (listen(fd, 4096) < 0) {
        fprintf(stderr, "[http_server] listen: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    G_SRV.listen_fd = fd;

    G_SRV.epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    if (G_SRV.epoll_fd < 0) {
        fprintf(stderr, "[http_server] epoll_create1: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    fprintf(stderr, "[http_server] listening on %s\n", sock_path);
    return 0;
}

// Spawn the event loop on a new thread; return immediately.
int32_t start_http_server(const char *sock_path) {
    if (setup_listener(sock_path) != 0) return -1;
    if (pthread_create(&G_SRV.thread, NULL, server_thread, NULL) != 0) {
        fprintf(stderr, "[http_server] pthread_create failed\n");
        return -1;
    }
    return 0;
}
// Run the event loop in the calling thread (blocks forever). Used when the
// caller wants its own thread to BE the event loop, so the JS runtime's
// GC/timer threads aren't competing for CPU with the I/O loop.
int32_t run_http_server(const char *sock_path) {
    if (setup_listener(sock_path) != 0) return -1;
    server_thread(NULL);
    return 0;
}
