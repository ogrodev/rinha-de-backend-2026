// API entrypoint. Binds `Bun.serve` IMMEDIATELY (so the port is reachable
// for liveness probes and the LB healthcheck) and starts the index load in
// the background. While the index is still loading, `/ready` returns 503 and
// `/fraud-score` returns 503 — see `handlers.ts`. When the index is loaded,
// `state.ready` flips to true and traffic is accepted.
//
// On load failure we `process.exit(1)` so docker restarts the container. The
// nginx LB has the APIs gated on healthchecks, so the warming window never
// produces user-visible 503s through `:9999`.

import { handleFraudScore, type AppState } from "./handlers.ts";
import { loadIndex } from "./index/load.ts";
import { makeScratch, bindIndex, searchFraudCount } from "./index/search.ts";

const DATA_DIR = process.env.DATA_DIR ?? "/app/data";
const PORT = Number(process.env.PORT ?? 8080);

const NOT_FOUND_BODY = '{"error":"not_found"}';
const JSON_HEADERS = { "content-type": "application/json" } as const;

const state: AppState = { ready: false };

// Pre-built static responses; we swap /ready from "not ready" to "ready"
// via server.reload() after the index loads + JIT warms up.
const READY_RESPONSE = new Response("{}", {
  status: 200,
  headers: { "content-type": "application/json" },
});
const NOT_READY_RESPONSE = new Response('{"error":"not_ready"}', {
  status: 503,
  headers: { "content-type": "application/json" },
});

const server = Bun.serve({
  port: PORT,
  // Bun 1.3 routes table — uWebSocket-style trie + JSC structure cache.
  routes: {
    "/ready": NOT_READY_RESPONSE,
    "/fraud-score": {
      POST: (req) => handleFraudScore(req, state),
    },
  },
  fetch() {
    return new Response(NOT_FOUND_BODY, { status: 404, headers: JSON_HEADERS });
  },
});

console.error(`[server] listening on :${server.port} (DATA_DIR=${DATA_DIR})`);

// Background index load. Bun.serve is already accepting connections; until we
// finish, `/ready` returns 503.
loadIndex(DATA_DIR)
  .then((idx) => {
    state.idx = idx;
    state.scratch = makeScratch(idx.k, idx.nprobe);
    state.queryBuf = new Float32Array(idx.d);
    bindIndex(idx);

    // JIT warmup: run a few thousand searches synchronously before flipping
    // ready, so the very first k6 ramp requests don't pay V8 compilation
    // overhead at p99. ~150ms total.
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
