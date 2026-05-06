// API entrypoint. Binds `Bun.serve` on a Unix domain socket for low-latency
// IPC with the SoNoForevis Rust load balancer. The LB on the same host
// connects to /run/sock/api{1,2}.sock and forwards bytes — no TCP loopback,
// no HTTP-aware proxy.
//
// While the index is still loading, /ready returns 503 (via initial route
// table) and /fraud-score returns 503 (via the handler's state.ready check).
// Once the index loads + JIT warms up, /ready is swapped to a static 200 {}
// via server.reload() and state.ready flips to true.
//
// On load failure we exit(1) so docker restarts the container. The LB is
// gated on each API's healthcheck, so the warming window never produces
// user-visible 503s through :9999.

import fs from "node:fs";
import { handleFraudScore, type AppState } from "./handlers.ts";
import { loadIndex } from "./index/load.ts";
import { makeScratch, bindIndex, searchFraudCount } from "./index/search.ts";

const DATA_DIR = process.env.DATA_DIR ?? "/app/data";
// Listening mode: UDS in production (SOCK set in Dockerfile), TCP in tests
// (PORT set by the test harness). If both are set, UDS wins.
const SOCK = process.env.SOCK;
const PORT = SOCK ? null : Number(process.env.PORT ?? 8080);

const NOT_FOUND_BODY = '{"error":"not_found"}';
const JSON_HEADERS = { "content-type": "application/json" } as const;

const state: AppState = { ready: false };

// Pre-built static responses — swapped via server.reload() once warm.
const READY_RESPONSE = new Response("{}", {
  status: 200,
  headers: { "content-type": "application/json" },
});
const NOT_READY_RESPONSE = new Response('{"error":"not_ready"}', {
  status: 503,
  headers: { "content-type": "application/json" },
});

// Stale socket cleanup before listen. Skipped when listening on TCP.
if (SOCK) {
  try { fs.unlinkSync(SOCK); } catch {}
}

// Construct ServeOptions conditionally so TS narrows the union correctly.
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

// Make the UDS world-writable so the LB container (different uid) can
// connect.
if (SOCK) {
  try { fs.chmodSync(SOCK, 0o666); } catch {}
}

console.error(
  `[server] listening on ${SOCK ? SOCK : ":" + PORT} (DATA_DIR=${DATA_DIR})`,
);

// Background index load. Bun.serve already accepts connections; /ready=503
// while loading.
loadIndex(DATA_DIR)
  .then((idx) => {
    state.idx = idx;
    state.scratch = makeScratch(idx.k, idx.nprobe);
    state.queryBuf = new Float32Array(idx.d);
    bindIndex(idx);

    // JIT warmup: 50k searches before flipping ready.
    const warmupQuery = new Float32Array(idx.d);
    for (let dim = 0; dim < idx.d; dim++) {
      warmupQuery[dim] = (idx.vectors[dim] as number) * (idx.decodeFactor[dim] as number);
    }
    for (let i = 0; i < 50_000; i++) {
      searchFraudCount(idx, warmupQuery, state.scratch);
    }

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
    console.error(
      `[server] ready: n=${idx.n} k=${idx.k} d=${idx.d} nprobe=${idx.nprobe}`,
    );
  })
  .catch((err) => {
    console.error("[server] index load failed:", err);
    process.exit(1);
  });

// Clean up the socket on shutdown.
function shutdown(): void {
  if (SOCK) {
    try { fs.unlinkSync(SOCK); } catch {}
  }
  process.exit(0);
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
process.on("SIGHUP", shutdown);
