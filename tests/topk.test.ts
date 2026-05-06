import { describe, expect, test } from "bun:test";
import { topKInit, topKConsider } from "../src/util/topk.ts";

const SENTINEL_IDX = 0xFFFFFFFF;

function makeBuf(k: number) {
  const d = new Float32Array(k);
  const i = new Uint32Array(k);
  topKInit(d, i);
  return { d, i };
}

describe("topK insertion sort", () => {
  test("init fills with +Infinity and 0xFFFFFFFF", () => {
    const { d, i } = makeBuf(5);
    for (let j = 0; j < 5; j++) {
      expect(d[j]).toBe(Number.POSITIVE_INFINITY);
      expect(i[j]).toBe(SENTINEL_IDX);
    }
  });

  test("single insert lands at [0]", () => {
    const { d, i } = makeBuf(5);
    topKConsider(d, i, 1.5, 42);
    expect(d[0]).toBe(1.5);
    expect(i[0]).toBe(42);
    expect(d[1]).toBe(Number.POSITIVE_INFINITY);
    expect(i[1]).toBe(SENTINEL_IDX);
  });

  test("k inserts come out ascending", () => {
    const { d, i } = makeBuf(5);
    const inputs: Array<[number, number]> = [
      [3.0, 30], [1.0, 10], [4.0, 40], [2.0, 20], [5.0, 50],
    ];
    for (const [dist, idx] of inputs) topKConsider(d, i, dist, idx);
    expect(Array.from(d)).toEqual([1.0, 2.0, 3.0, 4.0, 5.0]);
    expect(Array.from(i)).toEqual([10, 20, 30, 40, 50]);
  });

  test("k+m inserts drop the largest", () => {
    const { d, i } = makeBuf(3);
    // Five entries, only top-3 (smallest) survive.
    topKConsider(d, i, 5.0, 50);
    topKConsider(d, i, 1.0, 10);
    topKConsider(d, i, 9.0, 90); // dropped
    topKConsider(d, i, 2.0, 20);
    topKConsider(d, i, 7.0, 70); // dropped
    expect(Array.from(d)).toEqual([1.0, 2.0, 5.0]);
    expect(Array.from(i)).toEqual([10, 20, 50]);
  });

  test("ties keep first-inserted index (stability)", () => {
    const { d, i } = makeBuf(3);
    topKConsider(d, i, 1.0, 100);
    topKConsider(d, i, 1.0, 200); // same distance, later
    topKConsider(d, i, 1.0, 300); // same distance, latest
    // Stable insertion: equal-distance entries keep insertion order
    expect(Array.from(d)).toEqual([1.0, 1.0, 1.0]);
    expect(Array.from(i)).toEqual([100, 200, 300]);
  });

  test("worse-than-worst is rejected without writing", () => {
    const { d, i } = makeBuf(3);
    topKConsider(d, i, 1.0, 10);
    topKConsider(d, i, 2.0, 20);
    topKConsider(d, i, 3.0, 30);
    topKConsider(d, i, 99.0, 999);
    expect(Array.from(d)).toEqual([1.0, 2.0, 3.0]);
    expect(Array.from(i)).toEqual([10, 20, 30]);
  });

  test("zero allocation across 100k inserts", () => {
    const { d, i } = makeBuf(5);
    if (typeof Bun !== "undefined" && (Bun as any).gc) (Bun as any).gc(true);
    const before = process.memoryUsage().heapUsed;
    for (let n = 0; n < 100_000; n++) {
      // Pseudo-random distance in [0,1)
      const x = Math.sin(n * 12.9898) * 43758.5453;
      const dist = x - Math.floor(x);
      topKConsider(d, i, dist, n);
    }
    const after = process.memoryUsage().heapUsed;
    expect(after - before).toBeLessThan(1024 * 1024);
  });
});
