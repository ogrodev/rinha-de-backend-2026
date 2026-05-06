// HTTP handlers for `/ready` and `/fraud-score`.
//
// Two hot-path implementations:
//   1. Native (FFI loaded): single C call processes the entire pipeline —
//      JSON parse, vectorize, search, return fraud count. Saves a JS-side
//      JSON.parse and one FFI cross. Used in production.
//   2. JS fallback: parses with req.json(), vectorizes in TS, calls
//      searchFraudCount which dispatches to FFI for search alone. Used by
//      the test suite where the C lib isn't compiled.
//
// Response bodies for the 6 possible fraud scores are pre-encoded as
// Uint8Array buffers at module init.

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

export function handleReady(state: AppState): Response {
  return state.ready
    ? new Response(READY_BODY, { status: 200, headers: JSON_HEADERS })
    : new Response(NOT_READY_BODY, { status: 503, headers: JSON_HEADERS });
}

export async function handleFraudScore(
  req: Request,
  state: AppState,
): Promise<Response> {
  if (!state.ready || !state.idx || !state.scratch || !state.queryBuf) {
    return new Response(NOT_READY_BODY, { status: 503, headers: JSON_HEADERS });
  }

  // Fast path — entire pipeline in C.
  if (FFI_LOADED) {
    try {
      const bytes = await req.bytes();
      const count = processRequest(bytes, state.queryBuf);
      if (count < 0 || count > 5) {
        return new Response(INVALID_PAYLOAD_BODY, { status: 400, headers: JSON_HEADERS });
      }
      return new Response(SCORE_BODIES[count], { status: 200, headers: JSON_HEADERS });
    } catch {
      return new Response(INTERNAL_BODY, { status: 500, headers: JSON_HEADERS });
    }
  }

  // JS fallback path (tests).
  try {
    let body: any;
    try {
      body = await req.json();
    } catch {
      return new Response(INVALID_JSON_BODY, { status: 400, headers: JSON_HEADERS });
    }
    const idx = state.idx;
    const ok = vectorize(body, idx.norm, idx.mccRisk, state.queryBuf);
    if (!ok) {
      return new Response(INVALID_PAYLOAD_BODY, { status: 400, headers: JSON_HEADERS });
    }
    const fraudCount = searchFraudCount(idx, state.queryBuf, state.scratch);
    return new Response(SCORE_BODIES[fraudCount] ?? SCORE_BODIES[0]!, { status: 200, headers: JSON_HEADERS });
  } catch {
    return new Response(INTERNAL_BODY, { status: 500, headers: JSON_HEADERS });
  }
}
