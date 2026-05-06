// Bun.listen-based custom HTTP/1.1 server.
//
// Bun.serve has per-request overhead (Request/Response/Headers/URL allocation,
// async function boundary) that hits ~50ms p99 on the contest rig under
// 900 RPS. Bun.listen gives us raw TCP and lets the C library handle HTTP
// parsing + business logic + response building in a single FFI call —
// eliminating per-request JS allocation almost entirely.
//
// Per-connection state is small (req-accumulator buffer + write pointer).
// Per-request: zero allocation in the hot path; the response is written
// directly from a static C buffer via socket.write(Uint8Array).

import { dlopen, FFIType, ptr, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { loadIndex } from "./index/load.ts";
import { bindIndex, searchFraudCount, makeScratch } from "./index/search.ts";

const DATA_DIR = process.env.DATA_DIR ?? "/app/data";
const PORT = Number(process.env.PORT ?? 8080);

// --- FFI binding -----------------------------------------------------------

function findLib(): string | null {
  const candidates = [
    process.env.SEARCH_LIB,
    `/app/libsearch.${suffix}`,
    `./libsearch.${suffix}`,
    join(dirname(import.meta.dir), `libsearch.${suffix}`),
    join(dirname(import.meta.dir), `native/libsearch.${suffix}`),
  ].filter((p): p is string => typeof p === "string" && p.length > 0);
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  return null;
}

const libPath = findLib();
if (!libPath) {
  console.error("[server-listen] libsearch shared library not found");
  process.exit(1);
}
console.error(`[server-listen] using ${libPath}`);

const ffi = dlopen(libPath, {
  set_ready: { args: [FFIType.i32], returns: FFIType.void },
  handle_http: {
    args: [FFIType.ptr, FFIType.i32, FFIType.ptr, FFIType.i32, FFIType.ptr],
    returns: FFIType.i32,
  },
  // Also load the index-init symbols (search_init etc.) so bindIndex works.
}).symbols;

// --- Per-connection state --------------------------------------------------

const REQ_BUF_INITIAL = 4 * 1024;   // 4 KB; grown on demand
const REQ_BUF_MAX = 64 * 1024;
const RESP_BUF_SIZE = 4 * 1024;     // largest possible response is ~120 bytes

type ConnState = {
  req: Uint8Array;
  reqLen: number;          // bytes of `req` currently used
  resp: Uint8Array;        // pinned response scratch
  query: Float32Array;     // pinned query scratch (passed to handle_http)
};

// Pre-allocated pool of scratch buffers, reused across connections.
// Connections come and go; we recycle to avoid GC churn.
const POOL: ConnState[] = [];
function acquireState(): ConnState {
  const s = POOL.pop();
  if (s) {
    s.reqLen = 0;
    return s;
  }
  return {
    req: new Uint8Array(REQ_BUF_INITIAL),
    reqLen: 0,
    resp: new Uint8Array(RESP_BUF_SIZE),
    query: new Float32Array(14),
  };
}
function releaseState(s: ConnState): void {
  if (POOL.length < 256) POOL.push(s);
}

// --- Index load + warmup ---------------------------------------------------

const idx = await loadIndex(DATA_DIR);
bindIndex(idx);

// JIT warmup. Call searchFraudCount many times so the FFI shim is hot.
const warmupQuery = new Float32Array(idx.d);
for (let dim = 0; dim < idx.d; dim++) {
  warmupQuery[dim] = (idx.vectors[dim] as number) * (idx.decodeFactor[dim] as number);
}
const warmupScratch = makeScratch(idx.k, idx.nprobe);
for (let i = 0; i < 50_000; i++) {
  searchFraudCount(idx, warmupQuery, warmupScratch);
}

ffi.set_ready(1);
console.error(`[server-listen] ready: n=${idx.n} k=${idx.k} d=${idx.d} nprobe=${idx.nprobe}`);

// --- TCP server ------------------------------------------------------------

Bun.listen<ConnState>({
  hostname: "0.0.0.0",
  port: PORT,
  socket: {
    open(socket) {
      socket.data = acquireState();
    },
    data(socket, chunk) {
      const s = socket.data;
      // Grow req buffer if needed.
      if (s.reqLen + chunk.byteLength > s.req.byteLength) {
        const need = s.reqLen + chunk.byteLength;
        if (need > REQ_BUF_MAX) {
          // Refuse oversized requests.
          socket.end();
          return;
        }
        let cap = s.req.byteLength;
        while (cap < need) cap *= 2;
        const grow = new Uint8Array(cap);
        grow.set(s.req.subarray(0, s.reqLen));
        s.req = grow;
      }
      s.req.set(chunk, s.reqLen);
      s.reqLen += chunk.byteLength;

      // Try to drain as many complete requests as possible (HTTP/1.1
      // pipelining). handle_http returns the consumed-vs-output via length:
      // the C parser walks from req[0]; on success, the whole request up to
      // body_end was consumed. We compute consumed length here in JS by
      // re-parsing Content-Length the same way (only when handle_http
      // succeeds), then shift the buffer.
      while (s.reqLen > 0) {
        const written = ffi.handle_http(
          ptr(s.req),
          s.reqLen,
          ptr(s.resp),
          s.resp.byteLength,
          ptr(s.query),
        );
        if (written === 0) break; // need more bytes
        if (written < 0) {
          socket.end();
          return;
        }
        // Determine how many bytes the C handler consumed.
        const consumed = consumedBytes(s.req, s.reqLen);
        if (consumed <= 0) {
          // Shouldn't happen if handle_http returned > 0; defensive.
          socket.end();
          return;
        }
        // Write the response.
        socket.write(s.resp.subarray(0, written));
        // Shift remaining bytes to the front.
        if (consumed === s.reqLen) {
          s.reqLen = 0;
        } else {
          s.req.copyWithin(0, consumed, s.reqLen);
          s.reqLen -= consumed;
        }
      }
    },
    close(socket) {
      releaseState(socket.data);
    },
    error(socket, err) {
      console.error("[server-listen] socket error:", err);
      releaseState(socket.data);
    },
  },
});
console.error(`[server-listen] listening on :${PORT}`);

// Compute how many bytes constituted one complete HTTP/1.1 request at the
// front of the buffer. Mirrors find_body_start + Content-Length parsing in
// process.c. Called only after handle_http said "I produced a response".
function consumedBytes(buf: Uint8Array, len: number): number {
  // Find \r\n\r\n
  for (let i = 0; i + 3 < len; i++) {
    if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) {
      const bodyStart = i + 4;
      // GET requests have no body — 1 request consumed.
      if (buf[0] === 71 /* G */) return bodyStart;
      // POST: read Content-Length.
      const cl = parseContentLength(buf, bodyStart);
      if (cl < 0) return -1;
      return bodyStart + cl;
    }
  }
  return -1;
}

function parseContentLength(buf: Uint8Array, headersEnd: number): number {
  // Look for "\r\nContent-Length:" or "\r\ncontent-length:".
  for (let i = 0; i + 17 < headersEnd; i++) {
    if (buf[i] !== 13 || buf[i + 1] !== 10) continue;
    // Compare 15 bytes of header name (case-insensitive).
    const ok =
      (matches(buf, i + 2, "Content-Length:") ||
        matches(buf, i + 2, "content-length:"));
    if (!ok) continue;
    let j = i + 17;
    while (j < headersEnd && buf[j] === 32) j++;
    let v = 0;
    while (j < headersEnd && buf[j]! >= 48 && buf[j]! <= 57) {
      v = v * 10 + (buf[j]! - 48);
      j++;
    }
    return v;
  }
  return -1;
}

function matches(buf: Uint8Array, off: number, needle: string): boolean {
  for (let k = 0; k < needle.length; k++) {
    if (buf[off + k] !== needle.charCodeAt(k)) return false;
  }
  return true;
}
