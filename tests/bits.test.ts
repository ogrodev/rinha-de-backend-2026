import { describe, expect, test } from "bun:test";
import { getBit } from "../src/util/bits.ts";

describe("getBit (LSB-first packed)", () => {
  test("known 24-byte pattern at boundary indices", () => {
    // Build a 24-byte buffer (192 bits) with a deterministic pattern.
    // We pack arbitrary bit values and read them back via getBit.
    const N = 192;
    const expected = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      // Mixed pattern: pseudo-random but deterministic.
      expected[i] = (i * 37 + 11) & 1;
    }
    const buf = new Uint8Array(N >> 3);
    for (let i = 0; i < N; i++) {
      if (expected[i]) buf[i >> 3]! |= 1 << (i & 7);
    }

    // Spot-check the documented boundary indices.
    for (const i of [0, 7, 8, 15, 23, 191]) {
      expect(getBit(buf, i)).toBe(expected[i] as 0 | 1);
    }

    // And verify the full buffer round-trips.
    for (let i = 0; i < N; i++) {
      expect(getBit(buf, i)).toBe(expected[i] as 0 | 1);
    }
  });

  test("LSB-first ordering: bit 0 of buf[0] is bit-0 of the stream", () => {
    const buf = new Uint8Array([0b00000001, 0b10000000]);
    expect(getBit(buf, 0)).toBe(1);
    expect(getBit(buf, 1)).toBe(0);
    expect(getBit(buf, 7)).toBe(0);
    // Byte 1, bit 7 (i.e. global index 15) is the MSB of the second byte.
    expect(getBit(buf, 15)).toBe(1);
    expect(getBit(buf, 8)).toBe(0);
  });
});
