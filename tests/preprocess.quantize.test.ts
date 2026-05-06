import { describe, expect, test } from "bun:test";
import { buildScale, quantizeAll } from "../scripts/preprocess.ts";
import { decodeFactor } from "../src/index/quantize.ts";

const D = 14;

function makeFlat(n: number, perDimMax: number[]): Float32Array {
  // Synthesize n*D floats so that for dim d, max |v| == perDimMax[d] exactly,
  // distributed roughly uniformly across [-perDimMax[d], perDimMax[d]].
  const flat = new Float32Array(n * D);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) {
      const max = perDimMax[d] as number;
      // Sweep [-max, max] across the n rows, plant ±max at first/last rows.
      const t = n === 1 ? 0 : (2 * i) / (n - 1) - 1; // [-1, 1]
      flat[i * D + d] = max * t;
    }
  }
  return flat;
}

describe("buildScale", () => {
  test("returns per-dim max abs over n vectors", () => {
    const n = 100;
    const perDimMax = [
      1, 2, 3, 0.5, 10, 100, 0.001, 7,
      0.1, 1000, 0.25, 1.5, 0.3, 8,
    ];
    const flat = makeFlat(n, perDimMax);
    const scale = buildScale(flat, n, D);
    expect(scale.length).toBe(D);
    for (let d = 0; d < D; d++) {
      expect(scale[d]).toBeCloseTo(perDimMax[d] as number, 6);
    }
  });

  test("zero column yields scale 0 (callers must guard)", () => {
    const flat = new Float32Array(D); // single zero row
    const scale = buildScale(flat, 1, D);
    for (let d = 0; d < D; d++) expect(scale[d]).toBe(0);
  });
});

describe("quantizeAll", () => {
  test("produces Int16Array(n*d) and round-trips within scale[d]/32767", () => {
    const n = 100;
    const perDimMax = [
      1, 2, 3, 0.5, 10, 100, 0.001, 7,
      0.1, 1000, 0.25, 1.5, 0.3, 8,
    ];
    const flat = makeFlat(n, perDimMax);
    const scale = buildScale(flat, n, D);
    const q = quantizeAll(flat, scale, n, D);

    expect(q.length).toBe(n * D);
    expect(q).toBeInstanceOf(Int16Array);

    let maxErrPerDim = new Float64Array(D);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < D; d++) {
        const orig = flat[i * D + d] as number;
        const factor = decodeFactor(scale[d] as number);
        const recon = (q[i * D + d] as number) * factor;
        const err = Math.abs(orig - recon);
        if (err > (maxErrPerDim[d] as number)) maxErrPerDim[d] = err;
      }
    }
    for (let d = 0; d < D; d++) {
      const bound = (scale[d] as number) / 32767 + 1e-9;
      expect(maxErrPerDim[d] as number).toBeLessThanOrEqual(bound);
    }
  });
});
