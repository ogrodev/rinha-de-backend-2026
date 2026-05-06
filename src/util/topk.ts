// Fixed-size top-K via bounded insertion sort.
//
// Buffers `d` (distances) and `i` (indices) are caller-owned and **must** share
// the same length k. After `topKInit`:
//   d[0..k-1] === +Infinity
//   i[0..k-1] === 0xFFFFFFFF
// After any number of `topKConsider` calls, `d` is non-decreasing and `i[t]`
// is the candidate index that produced `d[t]`. Equal-distance candidates
// preserve insertion order (stable).
//
// Zero allocation per call. Callers in the hot path use the same `d` / `i`
// across millions of comparisons by re-running `topKInit` between queries.

const SENTINEL_IDX = 0xFFFFFFFF;

export function topKInit(d: Float32Array, i: Uint32Array): void {
  d.fill(Number.POSITIVE_INFINITY);
  i.fill(SENTINEL_IDX);
}

export function topKConsider(
  d: Float32Array,
  i: Uint32Array,
  dist: number,
  idx: number,
): void {
  const k = d.length;
  // Worst surviving entry sits at the tail. Stable comparison: only insert
  // when *strictly* smaller, so equal-distance entries keep insertion order.
  const tail = d[k - 1] as number;
  if (dist >= tail) return;

  // Shift entries with strictly-greater distance one slot right, stopping at
  // the first slot whose distance is <= `dist` (which preserves stability).
  let j = k - 1;
  while (j > 0 && (d[j - 1] as number) > dist) {
    d[j] = d[j - 1] as number;
    i[j] = i[j - 1] as number;
    j--;
  }
  d[j] = dist;
  i[j] = idx >>> 0;
}
