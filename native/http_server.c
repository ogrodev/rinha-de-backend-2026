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

// --- Per-connection lifecycle -----------------------------------------------

static conn_t *conn_new(int fd) {
    conn_t *c = (conn_t *)malloc(sizeof(conn_t));
    if (!c) return NULL;
    c->fd = fd;
    c->read_len = 0;
    c->write_len = 0;
    c->write_off = 0;
    c->read_buf = (char *)malloc(READ_BUF_SIZE);
    c->write_buf = (char *)malloc(WRITE_BUF_SIZE);
    c->query_buf = (float *)aligned_alloc(64, sizeof(float) * 16);
    if (!c->read_buf || !c->write_buf || !c->query_buf) {
        free(c->read_buf);
        free(c->write_buf);
        free(c->query_buf);
        free(c);
        return NULL;
    }
    memset(c->query_buf, 0, sizeof(float) * 16);
    return c;
}

static void conn_close(conn_t *c) {
    if (!c) return;
    if (c->fd >= 0) {
        epoll_ctl(G_SRV.epoll_fd, EPOLL_CTL_DEL, c->fd, NULL);
        close(c->fd);
    }
    free(c->read_buf);
    free(c->write_buf);
    free(c->query_buf);
    free(c);
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

        int32_t written = handle_http(
            c->read_buf, c->read_len,
            c->write_buf, WRITE_BUF_SIZE,
            c->query_buf
        );
        if (written == 0) return 0;
        if (written < 0)  return -1;

        // Find body start.
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
            consumed = body_start;
        } else {
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
        if (c->read_len < consumed) return 0;

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

int32_t start_http_server(const char *sock_path) {
    memset(&G_SRV, 0, sizeof(G_SRV));

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

    if (pthread_create(&G_SRV.thread, NULL, server_thread, NULL) != 0) {
        fprintf(stderr, "[http_server] pthread_create failed\n");
        close(G_SRV.epoll_fd);
        close(fd);
        return -1;
    }

    fprintf(stderr, "[http_server] listening on %s\n", sock_path);
    return 0;
}
