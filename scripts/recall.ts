// Sampled-reference recall@5 gate (spec §3.2 step 8).
//
// Compares IVF approximate search vs exact brute-force on the SAME index. We
// sample reference indices as query points, dequantize them on the fly, and
// run both methods. The "self-hit" at distance 0 is returned by both methods
// and cancels out of the recall metric — we don't need to special-case it.

// Self-contained inlined utilities so this script doesn't import from src/
// (keeps the docker build cache stable across runtime-only edits).
import { lcg } from "./kmeans.ts";

function topKInit(d: Float32Array, i: Uint32Array): void {
  d.fill(Number.POSITIVE_INFINITY);
  i.fill(0xFFFFFFFF);
}
function topKConsider(d: Float32Array, i: Uint32Array, dist: number, idx: number): void {
  const k = d.length;
  const tail = d[k - 1] as number;
  if (dist >= tail) return;
  let j = k - 1;
  while (j > 0 && (d[j - 1] as number) > dist) {
    d[j] = d[j - 1] as number;
    i[j] = i[j - 1] as number;
    j--;
  }
  d[j] = dist;
  i[j] = idx >>> 0;
}
function decodeFactor(scale: number): number {
  return scale / 32767;
}

// Scratch buffers reused across queries inside a single recallAt5 invocation.
type Scratch = {
  query: Float32Array;
  centroidDist: Float32Array;
  topProbeDist: Float32Array;
  topProbeIdx: Uint32Array;
  exactDist: Float32Array;
  exactIdx: Uint32Array;
  approxDist: Float32Array;
  approxIdx: Uint32Array;
  decodeFactors: Float32Array;
};

function dequantizeRowInto(
  vectors: Int16Array,
  rowIdx: number,
  decodeFactors: Float32Array,
  d: number,
  out: Float32Array,
): void {
  const base = rowIdx * d;
  for (let dim = 0; dim < d; dim++) {
    out[dim] = (vectors[base + dim] as number) * (decodeFactors[dim] as number);
  }
}

function squaredL2(
  query: Float32Array,
  vectors: Int16Array,
  rowIdx: number,
  decodeFactors: Float32Array,
  d: number,
): number {
  const base = rowIdx * d;
  let sum = 0;
  for (let dim = 0; dim < d; dim++) {
    const q = query[dim] as number;
    const v = (vectors[base + dim] as number) * (decodeFactors[dim] as number);
    const diff = q - v;
    sum += diff * diff;
  }
  return sum;
}

function squaredL2ToCentroid(
  query: Float32Array,
  centroids: Float32Array,
  c: number,
  d: number,
): number {
  const base = c * d;
  let sum = 0;
  for (let dim = 0; dim < d; dim++) {
    const diff = (query[dim] as number) - (centroids[base + dim] as number);
    sum += diff * diff;
  }
  return sum;
}

function exactTop5(
  query: Float32Array,
  vectors: Int16Array,
  decodeFactors: Float32Array,
  n: number,
  d: number,
  s: Scratch,
): void {
  topKInit(s.exactDist, s.exactIdx);
  for (let i = 0; i < n; i++) {
    const dist = squaredL2(query, vectors, i, decodeFactors, d);
    topKConsider(s.exactDist, s.exactIdx, dist, i);
  }
}

function ivfTop5(
  query: Float32Array,
  vectors: Int16Array,
  centroids: Float32Array,
  offsets: Uint32Array,
  decodeFactors: Float32Array,
  k: number,
  d: number,
  nprobe: number,
  s: Scratch,
): void {
  topKInit(s.topProbeDist, s.topProbeIdx);
  for (let c = 0; c < k; c++) {
    const cd = squaredL2ToCentroid(query, centroids, c, d);
    topKConsider(s.topProbeDist, s.topProbeIdx, cd, c);
  }
  topKInit(s.approxDist, s.approxIdx);
  for (let p = 0; p < nprobe; p++) {
    const c = s.topProbeIdx[p] as number;
    if (c === 0xFFFFFFFF) continue; // not enough centroids
    const lo = offsets[c] as number;
    const hi = offsets[c + 1] as number;
    for (let i = lo; i < hi; i++) {
      const dist = squaredL2(query, vectors, i, decodeFactors, d);
      topKConsider(s.approxDist, s.approxIdx, dist, i);
    }
  }
}

function intersectionSize5(a: Uint32Array, b: Uint32Array): number {
  // Both are length 5. Use a tiny set lookup.
  let count = 0;
  for (let i = 0; i < 5; i++) {
    const ai = a[i] as number;
    if (ai === 0xFFFFFFFF) continue;
    for (let j = 0; j < 5; j++) {
      if ((b[j] as number) === ai) {
        count++;
        break;
      }
    }
  }
  return count;
}

export function recallAt5(
  vectors: Int16Array,
  scale: Float32Array,
  offsets: Uint32Array,
  centroids: Float32Array,
  n: number,
  d: number,
  k: number,
  nprobe: number,
  sampleSize: number,
  seed: number,
): number {
  const decodeFactors = new Float32Array(d);
  for (let dim = 0; dim < d; dim++) decodeFactors[dim] = decodeFactor(scale[dim] as number);

  const s: Scratch = {
    query: new Float32Array(d),
    centroidDist: new Float32Array(k),
    topProbeDist: new Float32Array(nprobe),
    topProbeIdx: new Uint32Array(nprobe),
    exactDist: new Float32Array(5),
    exactIdx: new Uint32Array(5),
    approxDist: new Float32Array(5),
    approxIdx: new Uint32Array(5),
    decodeFactors,
  };

  const rng = lcg(seed);
  let totalIntersect = 0;
  for (let q = 0; q < sampleSize; q++) {
    let qi = Math.floor(rng() * n);
    if (qi >= n) qi = n - 1;
    dequantizeRowInto(vectors, qi, decodeFactors, d, s.query);

    exactTop5(s.query, vectors, decodeFactors, n, d, s);
    ivfTop5(s.query, vectors, centroids, offsets, decodeFactors, k, d, nprobe, s);

    totalIntersect += intersectionSize5(s.approxIdx, s.exactIdx);
  }
  return totalIntersect / (sampleSize * 5);
}
