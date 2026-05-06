// Symmetric int16 quantization helpers used by the build-time preprocessor and
// the runtime hot path.
//
// We use int16 (not int8) because the contest's labeling is brute-force k-NN
// on FLOAT vectors. int8 introduces enough rounding error in distance
// computations that ~0.2% of queries near tie boundaries land on a different
// top-5 set than the float ground truth, manifesting as FP/FN at scoring
// time. int16 cuts the per-dim quantization error by 256× — enough that
// quantization-induced ties become essentially nonexistent.
//
// Encoding (build-time): clamp(round(value * 32767 / scale), -32767, 32767).
// Decoding (hot path):   int16 * decodeFactor(scale) where decodeFactor =
//                        scale / 32767.
//
// NaN policy: callers MUST NOT pass NaN (vectorize / build-time parser
// guarantee finite floats).
//
// Round-trip error: |value - decodeFactor(scale) * encodeI16(value, scale)|
// is bounded by scale / 32767 for value in [-scale, scale].

export function encodeI16(value: number, scale: number): number {
  const q = Math.round((value * 32767) / scale);
  return q < -32767 ? -32767 : q > 32767 ? 32767 : q;
}

export function decodeFactor(scale: number): number {
  return scale / 32767;
}
