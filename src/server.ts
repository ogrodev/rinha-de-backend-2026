// API entrypoint.
//
// Two modes:
//   1. Production (libsearch.so loaded): the C HTTP server takes over the
//      socket entirely. Bun's role is reduced to:
//        - Load the index from disk (via Bun.file)
//        - Call bindIndex() to hand index pointers to C
//        - Run JIT warmup (still useful for the JS fallback paths)
//        - Call startHttpServer(SOCK) — spawns a C epoll thread
//        - Call setReady(true)
//        - Idle the main event loop forever
//      Bun.serve is NOT used; no fetch handler in the request path.
//
//   2. Tests / no FFI: fall back to Bun.serve (TCP via PORT env) so the
//      existing test harness keeps working.

import fs from "node:fs";
import { handleFraudScore, type AppState } from "./handlers.ts";
import { loadIndex } from "./index/load.ts";
import {
  makeScratch,
  bindIndex,
  searchFraudCount,
  startHttpServer,
  setReady,
  isFfiLoaded,
} from "./index/search.ts";

const DATA_DIR = process.env.DATA_DIR ?? "/app/data";
const SOCK = process.env.SOCK;
const PORT = SOCK ? null : Number(process.env.PORT ?? 8080);

const NOT_FOUND_BODY = '{"error":"not_found"}';
const JSON_HEADERS = { "content-type": "application/json" } as const;

const state: AppState = { ready: false };

async function main(): Promise<void> {
  console.error(
    `[server] starting (DATA_DIR=${DATA_DIR}, ${SOCK ? "SOCK=" + SOCK : "PORT=" + PORT})`,
  );

  const idx = await loadIndex(DATA_DIR);
  state.idx = idx;
  state.scratch = makeScratch(idx.k, idx.nprobe);
  state.queryBuf = new Float32Array(idx.d);
  bindIndex(idx);

  // JIT warmup. Still useful for JS fallback path; cheap on the C path.
  const warmupQuery = new Float32Array(idx.d);
  for (let dim = 0; dim < idx.d; dim++) {
    warmupQuery[dim] =
      (idx.vectors[dim] as number) * (idx.decodeFactor[dim] as number);
  }
  // Pre-warm pages: touch every 4KB of the big int16 vectors array so the
  // first request doesn't pay for demand page-faults (~100µs each on the
  // rig). The 84 MB array has ~21k pages; a single linear pass dirties them
  // all into resident memory.
  let touchSum = 0;
  const stride = 4096 / 2; // int16 elements per page
  for (let i = 0; i < idx.vectors.length; i += stride) {
    touchSum += (idx.vectors[i] as number);
  }
  // Same for centroids (much smaller but still helps).
  for (let i = 0; i < idx.centroids.length; i += 1024) {
    touchSum += (idx.centroids[i] as number);
  }
  // Force the optimizer to keep the touch (otherwise dead-code elimination
  // could skip the loop entirely).
  if (touchSum === Number.NaN) console.error("unreachable");

  for (let i = 0; i < 50_000; i++) {
    searchFraudCount(idx, warmupQuery, state.scratch);
  }

  if (isFfiLoaded() && SOCK) {
    // Native path: hand the socket to C and idle.
    if (fs.existsSync(SOCK)) {
      try { fs.unlinkSync(SOCK); } catch {}
    }
    if (!startHttpServer(SOCK)) {
      console.error("[server] startHttpServer failed");
      process.exit(1);
    }
    setReady(true);
    state.ready = true;
    console.error(
      `[server] native http server up: n=${idx.n} k=${idx.k} d=${idx.d} nprobe=${idx.nprobe}`,
    );
    // Hold the event loop open. The C thread does all the work.
    setInterval(() => {}, 1 << 30);
    return;
  }

  // Fallback path: Bun.serve on TCP (tests).
  const READY_RESPONSE = new Response("{}", {
    status: 200,
    headers: { "content-type": "application/json" },
  });
  const NOT_READY_RESPONSE = new Response('{"error":"not_ready"}', {
    status: 503,
    headers: { "content-type": "application/json" },
  });
  const baseOptions = {
    routes: {
      "/ready": NOT_READY_RESPONSE,
      "/fraud-score": {
        POST: (req: Request) => handleFraudScore(req, state),
      },
    },
    fetch() {
      return new Response(NOT_FOUND_BODY, { status: 404, headers: JSON_HEADERS });
    },
  };
  const server = SOCK
    ? Bun.serve({ unix: SOCK, ...baseOptions })
    : Bun.serve({ port: PORT!, ...baseOptions });
  if (SOCK) {
    try { fs.chmodSync(SOCK, 0o666); } catch {}
  }
  console.error(
    `[server] Bun.serve fallback on ${SOCK ?? ":" + PORT}`,
  );
  server.reload({
    routes: {
      "/ready": READY_RESPONSE,
      "/fraud-score": {
        POST: (req) => handleFraudScore(req, state),
      },
    },
    fetch() {
      return new Response(NOT_FOUND_BODY, { status: 404, headers: JSON_HEADERS });
    },
  });
  state.ready = true;
  console.error(`[server] ready (fallback)`);
}

function shutdown(): void {
  if (SOCK) {
    try { fs.unlinkSync(SOCK); } catch {}
  }
  process.exit(0);
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
process.on("SIGHUP", shutdown);

main().catch((err) => {
  console.error("[server] failed:", err);
  process.exit(1);
});
