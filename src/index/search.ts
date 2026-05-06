// IVF (Inverted File) approximate nearest-neighbor search over the cluster-
// sorted int16 vectors.
//
// Two implementations:
//   - Native SIMD via bun:ffi (NEON on aarch64, AVX2/FMA on x86_64) when
//     `libsearch.so` is loadable. This is the production hot path inside the
//     docker image.
//   - Pure-Bun JS fallback for tests and environments without the compiled
//     library. Algorithm matches the C version exactly so they're
//     byte-equivalent on a fixed dataset (modulo fp rounding order in SIMD).
//
// Both paths share the same `search(idx, query, scratch)` signature so
// callers don't care which is active.

import type { Index } from "./types.ts";
import { topKInit, topKConsider } from "../util/topk.ts";
import { getBit } from "../util/bits.ts";
import { dlopen, FFIType, ptr, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";

export type SearchScratch = {
  topProbeDist: Float32Array;
  topProbeIdx: Uint32Array;
  top5Dist: Float32Array;
  top5Idx: Uint32Array;
};

export function makeScratch(_k: number, nprobeMax: number): SearchScratch {
  return {
    topProbeDist: new Float32Array(nprobeMax),
    topProbeIdx: new Uint32Array(nprobeMax),
    top5Dist: new Float32Array(5),
    top5Idx: new Uint32Array(5),
  };
}

// --- FFI binding (loaded once at import) ---------------------------------------

type FfiSymbols = {
  search_init: (
    vectors: number, labels: number, centroids: number, offsets: number,
    radii: number, decodeFactor: number,
    n: number, k: number, d: number, nprobe: number,
  ) => void;
  search_query: (query: number) => number;
  // Single-call hot path: parses JSON body, vectorizes, searches, returns
  // fraud_count [0..5] (or -1 on parse error).
  process_request: (body: number, body_len: number, query_buf: number) => number;
};

function tryLoadNativeLib(): FfiSymbols | null {
  // Search order:
  //   1. SEARCH_LIB env (explicit override)
  //   2. /app/libsearch.<suffix>     (docker runtime layout)
  //   3. ./libsearch.<suffix>         (local cwd)
  //   4. <project root>/libsearch.<suffix>
  const candidates = [
    process.env.SEARCH_LIB,
    `/app/libsearch.${suffix}`,
    `./libsearch.${suffix}`,
    join(dirname(import.meta.dir), `../libsearch.${suffix}`),
  ].filter((p): p is string => typeof p === "string" && p.length > 0);

  for (const path of candidates) {
    try {
      if (!existsSync(path)) continue;
      const lib = dlopen(path, {
        search_init: {
          args: [
            FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.ptr,
            FFIType.i32, FFIType.i32, FFIType.i32, FFIType.i32,
          ],
          returns: FFIType.void,
        },
        search_query: {
          args: [FFIType.ptr],
          returns: FFIType.i32,
        },
        process_request: {
          args: [FFIType.ptr, FFIType.i32, FFIType.ptr],
          returns: FFIType.i32,
        },
      });
      console.error(`[search] using native FFI lib: ${path}`);
      return lib.symbols as unknown as FfiSymbols;
    } catch (err) {
      console.error(`[search] FFI candidate ${path} failed: ${(err as Error).message}`);
    }
  }
  return null;
}

const FFI: FfiSymbols | null = tryLoadNativeLib();

// Tracks the index pointers passed to `search_init` so subsequent
// `search_query` calls share the same buffers. We pin these on `bindIndex`
// so GC can't move them under us — the AppState already holds the originals,
// but ffi.ptr() needs the same TypedArray instance.
let bound: Index | null = null;

export function bindIndex(idx: Index): void {
  if (!FFI) {
    bound = idx;
    return;
  }
  FFI.search_init(
    ptr(idx.vectors),
    ptr(idx.labels),
    ptr(idx.centroids),
    ptr(idx.offsets),
    ptr(idx.radii),
    ptr(idx.decodeFactor),
    idx.n,
    idx.k,
    idx.d,
    idx.nprobe,
  );
  bound = idx;
}

// --- search ----------------------------------------------------------------

export function search(
  idx: Index,
  query: Float32Array,
  scratch: SearchScratch,
): number {
  if (FFI && bound === idx) {
    // Native path: returns fraud count directly.
    const frauds = FFI.search_query(ptr(query));
    return frauds / 5;
  }
  return searchJs(idx, query, scratch);
}

// --- JS fallback (used by tests; matches the C implementation) ---------------

function searchJs(
  idx: Index,
  query: Float32Array,
  scratch: SearchScratch,
): number {
  const D = idx.d;
  const k = idx.k;
  const nprobe = idx.nprobe;
  const centroids = idx.centroids;
  const offsets = idx.offsets;
  const vectors = idx.vectors;
  const decodeFactor = idx.decodeFactor;

  topKInit(scratch.topProbeDist, scratch.topProbeIdx);
  for (let c = 0; c < k; c++) {
    const cBase = c * D;
    let dist = 0;
    for (let dim = 0; dim < D; dim++) {
      const diff = (query[dim] as number) - (centroids[cBase + dim] as number);
      dist += diff * diff;
    }
    topKConsider(scratch.topProbeDist, scratch.topProbeIdx, dist, c);
  }

  topKInit(scratch.top5Dist, scratch.top5Idx);
  const radii = idx.radii;
  for (let p = 0; p < nprobe; p++) {
    const c = scratch.topProbeIdx[p] as number;
    if (c === 0xFFFFFFFF) continue;
    // Triangle-inequality cluster prune (mirrors native search.c).
    const dc = scratch.topProbeDist[p] as number;
    const rc = radii[c] as number;
    const sqrtDc = Math.sqrt(dc);
    if (sqrtDc > rc) {
      const diff = sqrtDc - rc;
      const lowerBound = diff * diff;
      if (lowerBound >= (scratch.top5Dist[4] as number)) continue;
    }
    const lo = offsets[c] as number;
    const hi = offsets[c + 1] as number;
    for (let i = lo; i < hi; i++) {
      const base = i * D;
      let dist = 0;
      for (let dim = 0; dim < D; dim++) {
        const v = (vectors[base + dim] as number) * (decodeFactor[dim] as number);
        const diff = (query[dim] as number) - v;
        dist += diff * diff;
      }
      topKConsider(scratch.top5Dist, scratch.top5Idx, dist, i);
    }
  }

  let frauds = 0;
  const labels = idx.labels;
  for (let t = 0; t < 5; t++) {
    const idxRow = scratch.top5Idx[t] as number;
    if (idxRow === 0xFFFFFFFF) continue;
    frauds += getBit(labels, idxRow);
  }
  return frauds / 5;
}

// Direct fraud-count path (skips the score→count→score round trip in the
// hot path). Used by handlers.ts which looks up a precomputed response body
// by fraud count (0..5).
export function searchFraudCount(
  idx: Index,
  query: Float32Array,
  scratch: SearchScratch,
): number {
  if (FFI && bound === idx) {
    return FFI.search_query(ptr(query));
  }
  // JS fallback: reuse search() but multiply back to count.
  return Math.round(searchJs(idx, query, scratch) * 5);
}

// Full request pipeline in C: parse JSON body, vectorize, search, return
// fraud count. Returns -1 if the body is malformed (caller returns 400).
export function processRequest(body: Uint8Array, queryBuf: Float32Array): number {
  if (!FFI) return -2; // FFI not loaded; caller falls back to JS path
  return FFI.process_request(ptr(body), body.length, ptr(queryBuf));
}

export function isFfiLoaded(): boolean {
  return FFI !== null;
}
