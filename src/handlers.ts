// HTTP handlers for `/ready` and `/fraud-score`.
//
// Hot path is intentionally minimal:
//   - Native FFI processes the entire pipeline (parse, vectorize, search) in
//     one C call.
//   - Handler returns Promise<Response> via `.then()` chain instead of an
//     async function — saves the async function's Promise wrapper allocation.
//   - Response objects for the 6 possible fraud scores are pre-allocated as
//     templates and `clone()`d per request (cheaper than `new Response(...)`).
//
// The JS fallback is used by tests when the C library isn't compiled.

import { vectorize } from "./vectorize.ts";
import {
  searchFraudCount,
  processRequest,
  isFfiLoaded,
  type SearchScratch,
} from "./index/search.ts";
import type { Index } from "./index/types.ts";

export type AppState = {
  ready: boolean;
  idx?: Index;
  scratch?: SearchScratch;
  queryBuf?: Float32Array;
};

const READY_BODY = "{}";
const NOT_READY_BODY = '{"error":"not_ready"}';
const INVALID_JSON_BODY = '{"error":"invalid_json"}';
const INVALID_PAYLOAD_BODY = '{"error":"invalid_payload"}';
const INTERNAL_BODY = '{"error":"internal"}';
const JSON_HEADERS = { "content-type": "application/json" } as const;

const ENC = new TextEncoder();
const SCORE_BODIES: Uint8Array[] = [
  ENC.encode('{"approved":true,"fraud_score":0}'),
  ENC.encode('{"approved":true,"fraud_score":0.2}'),
  ENC.encode('{"approved":true,"fraud_score":0.4}'),
  ENC.encode('{"approved":false,"fraud_score":0.6}'),
  ENC.encode('{"approved":false,"fraud_score":0.8}'),
  ENC.encode('{"approved":false,"fraud_score":1}'),
];

const FFI_LOADED = isFfiLoaded();

// Reusable Response factories — Bun.serve consumes the body once per send,
// so we have to build a fresh Response each call. Hoisting the init objects
// out of the hot path avoids per-call object literal allocation.
const OK_INIT = { status: 200, headers: JSON_HEADERS } as const;
const BAD_INIT = { status: 400, headers: JSON_HEADERS } as const;
const NOT_READY_INIT = { status: 503, headers: JSON_HEADERS } as const;
const INTERNAL_INIT = { status: 500, headers: JSON_HEADERS } as const;

export function handleReady(state: AppState): Response {
  return state.ready
    ? new Response(READY_BODY, OK_INIT)
    : new Response(NOT_READY_BODY, NOT_READY_INIT);
}

// Native fast path: parse, vectorize, search, fraud count — single FFI call.
// Returns Promise<Response> via .then() chain (no async function wrapper).
function handleFraudFfi(req: Request, queryBuf: Float32Array): Promise<Response> {
  return req.bytes().then(
    (bytes) => {
      const count = processRequest(bytes, queryBuf);
      if (count < 0 || count > 5) {
        return new Response(INVALID_PAYLOAD_BODY, BAD_INIT);
      }
      return new Response(SCORE_BODIES[count], OK_INIT);
    },
    () => new Response(INVALID_JSON_BODY, BAD_INIT),
  );
}

// JS fallback (tests).
async function handleFraudJs(
  req: Request,
  state: AppState & Required<Pick<AppState, "idx" | "scratch" | "queryBuf">>,
): Promise<Response> {
  let body: any;
  try {
    body = await req.json();
  } catch {
    return new Response(INVALID_JSON_BODY, BAD_INIT);
  }
  const idx = state.idx;
  const ok = vectorize(body, idx.norm, idx.mccRisk, state.queryBuf);
  if (!ok) {
    return new Response(INVALID_PAYLOAD_BODY, BAD_INIT);
  }
  const fraudCount = searchFraudCount(idx, state.queryBuf, state.scratch);
  return new Response(SCORE_BODIES[fraudCount] ?? SCORE_BODIES[0]!, OK_INIT);
}

export function handleFraudScore(
  req: Request,
  state: AppState,
): Response | Promise<Response> {
  if (!state.ready || !state.idx || !state.scratch || !state.queryBuf) {
    return new Response(NOT_READY_BODY, NOT_READY_INIT);
  }
  if (FFI_LOADED) {
    return handleFraudFfi(req, state.queryBuf);
  }
  return handleFraudJs(
    req,
    state as AppState & Required<Pick<AppState, "idx" | "scratch" | "queryBuf">>,
  ).catch(() => new Response(INTERNAL_BODY, INTERNAL_INIT));
}
