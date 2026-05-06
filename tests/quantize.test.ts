import { describe, expect, test } from "bun:test";
import { encodeI16, decodeFactor } from "../src/index/quantize.ts";

const Q = 32767;

describe("encodeI16 / decodeFactor", () => {
  test("decodeFactor is scale/32767", () => {
    expect(decodeFactor(1)).toBeCloseTo(1 / Q, 12);
    expect(decodeFactor(2)).toBeCloseTo(2 / Q, 12);
    expect(decodeFactor(0.5)).toBeCloseTo(0.5 / Q, 12);
  });

  test("round-trip max abs error <= scale/32767 for values in [-scale, scale]", () => {
    const scale = 1;
    const f = decodeFactor(scale);
    let maxErr = 0;
    for (let n = 0; n < 65535; n++) {
      const v = -1 + (2 * n) / 65534; // [-1, 1]
      const q = encodeI16(v, scale);
      const r = q * f;
      const err = Math.abs(v - r);
      if (err > maxErr) maxErr = err;
    }
    expect(maxErr).toBeLessThanOrEqual(scale / Q + 1e-9);
  });

  test("clamps to +32767 just above scale", () => {
    expect(encodeI16(1.0001, 1)).toBe(32767);
    expect(encodeI16(2.0, 1)).toBe(32767);
  });

  test("clamps to -32767 just below -scale", () => {
    expect(encodeI16(-1.0001, 1)).toBe(-32767);
    expect(encodeI16(-2.0, 1)).toBe(-32767);
  });

  test("zero encodes to 0", () => {
    expect(encodeI16(0, 1)).toBe(0);
    expect(encodeI16(0, 5.7)).toBe(0);
  });

  test("near-symmetric around zero", () => {
    expect(Math.abs(encodeI16(0.5, 1) - -encodeI16(-0.5, 1))).toBeLessThanOrEqual(1);
    expect(Math.abs(encodeI16(0.123, 1) - -encodeI16(-0.123, 1))).toBeLessThanOrEqual(1);
  });
});
