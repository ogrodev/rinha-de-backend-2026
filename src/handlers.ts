// HTTP handlers for `/ready` and `/fraud-score`.
//
// Hot path is intentionally minimal: a single try/catch wraps the JSON parse
// and the search call. Shape validation is implicit — vectorize() returns
// false on parse errors, and any field-access TypeError on a malformed
// payload falls through to the outer catch as a 500. The contest's payload
// generator emits well-formed JSON so this is fine in practice and saves
// ~5µs per request vs explicit per-field checks.
//
// Response bodies for the 6 possible fraud scores are pre-encoded as
// Uint8Array buffers at module init so the hot path doesn't pay for string
// concatenation per request.

import { vectorize } from "./vectorize.ts";
import { searchFraudCount, type SearchScratch } from "./index/search.ts";
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

// Precomputed score-response bodies. Index = fraud_count (0..5).
const ENC = new TextEncoder();
const SCORE_BODIES: Uint8Array[] = [
  ENC.encode('{"approved":true,"fraud_score":0}'),
  ENC.encode('{"approved":true,"fraud_score":0.2}'),
  ENC.encode('{"approved":true,"fraud_score":0.4}'),
  ENC.encode('{"approved":false,"fraud_score":0.6}'),
  ENC.encode('{"approved":false,"fraud_score":0.8}'),
  ENC.encode('{"approved":false,"fraud_score":1}'),
];

export function handleReady(state: AppState): Response {
  return state.ready
    ? new Response(READY_BODY, { status: 200, headers: JSON_HEADERS })
    : new Response(NOT_READY_BODY, { status: 503, headers: JSON_HEADERS });
}

export async function handleFraudScore(
  req: Request,
  state: AppState,
): Promise<Response> {
  // 503 BEFORE any body parse — warming-window probes never read the body.
  if (!state.ready || !state.idx || !state.scratch || !state.queryBuf) {
    return new Response(NOT_READY_BODY, { status: 503, headers: JSON_HEADERS });
  }

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

    // searchFraudCount returns 0..5 directly (FFI path bypasses score round-trip).
    const fraudCount = searchFraudCount(idx, state.queryBuf, state.scratch);
    const body_ = SCORE_BODIES[fraudCount] ?? SCORE_BODIES[0]!;
    return new Response(body_, { status: 200, headers: JSON_HEADERS });
  } catch {
    return new Response(INTERNAL_BODY, { status: 500, headers: JSON_HEADERS });
  }
}
