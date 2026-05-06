import { describe, expect, test } from "bun:test";
import { clamp01 } from "../src/util/clamp.ts";

describe("clamp01", () => {
  test("clamps negatives to 0", () => {
    expect(clamp01(-1)).toBe(0);
  });

  test("0 stays 0", () => {
    expect(clamp01(0)).toBe(0);
  });

  test("0.5 stays 0.5", () => {
    expect(clamp01(0.5)).toBe(0.5);
  });

  test("1 stays 1", () => {
    expect(clamp01(1)).toBe(1);
  });

  test("clamps values >1 to 1", () => {
    expect(clamp01(1.5)).toBe(1);
  });

  test("clamps +Infinity to 1", () => {
    expect(clamp01(Number.POSITIVE_INFINITY)).toBe(1);
  });

  test("clamps -Infinity to 0", () => {
    expect(clamp01(Number.NEGATIVE_INFINITY)).toBe(0);
  });

  test("propagates NaN unchanged", () => {
    expect(Number.isNaN(clamp01(Number.NaN))).toBe(true);
  });
});
