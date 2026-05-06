// Clamp value into [0, 1].
//
// Behavior contract (spec §6.1, §6.2):
//   - finite x in [0, 1]      -> x
//   - x < 0 (incl. -Infinity) -> 0
//   - x > 1 (incl. +Infinity) -> 1
//   - NaN                     -> NaN  (deliberate; caller treats as payload error)
//
// The spec wants `customer.avg_amount === 0` to land at 1.0 via clamp01(Infinity);
// it also wants any `NaN` dimension to be detected and rejected as
// `400 invalid_payload`. We propagate NaN here so post-vectorize callers
// can scan the buffer for NaNs once, instead of branching on every dim.
export function clamp01(x: number): number {
  // Self-inequality identifies NaN without extra calls (`x !== x`).
  return x !== x ? x : x < 0 ? 0 : x > 1 ? 1 : x;
}
