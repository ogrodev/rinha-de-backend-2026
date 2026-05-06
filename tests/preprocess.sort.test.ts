import { describe, expect, test } from "bun:test";
import { clusterSort } from "../scripts/preprocess.ts";
import { getBit } from "../src/util/bits.ts";

describe("clusterSort", () => {
  test("rows and labels reorder so each cluster is contiguous", () => {
    const N = 20;
    const D = 14;
    const K = 3;
    // Build a deterministic int8 dataset: row `i` has all int8 values equal to i.
    const vectors = new Int16Array(N * D);
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < D; d++) vectors[i * D + d] = (i % 127) as number;
    }
    // Pack labels: bit i = i mod 2.
    const labels = new Uint8Array(Math.ceil(N / 8));
    for (let i = 0; i < N; i++) {
      if (i % 2 === 0) labels[i >> 3]! |= 1 << (i & 7);
    }
    // Hand-built assignments.
    const assignments = new Uint32Array(N);
    for (let i = 0; i < N; i++) assignments[i] = (i * 7) % K;

    const { sortedVectors, sortedLabels, offsets } = clusterSort(
      vectors,
      labels,
      assignments,
      N,
      D,
      K,
    );

    expect(offsets.length).toBe(K + 1);
    expect(offsets[K]).toBe(N);
    // Sizes match per-cluster counts.
    const counts = [0, 0, 0];
    for (let i = 0; i < N; i++) counts[assignments[i] as number]!++;
    for (let c = 0; c < K; c++) {
      expect((offsets[c + 1] as number) - (offsets[c] as number)).toBe(counts[c] as number);
    }

    // For each destination row, verify (a) the source row's vector is preserved
    // (we identify by the unique "all-equal" pattern == original i mod 127),
    // and (b) the label bit at the destination matches the label bit at the
    // source.
    for (let dest = 0; dest < N; dest++) {
      // Recover the source-row identity: byte 0 of this row equals i mod 127.
      const id = sortedVectors[dest * D] as number;
      // Find which original i could produce that — but i mod 127 is ambiguous
      // only beyond i=127. We have N=20 < 127, so id IS the source i.
      const srcI = id;
      // Verify all D dims match.
      for (let d = 0; d < D; d++) {
        expect(sortedVectors[dest * D + d]).toBe(srcI as number);
      }
      // Verify the label bit at `dest` equals the original label bit at `srcI`.
      const expectedBit = (srcI % 2 === 0 ? 1 : 0) as 0 | 1;
      expect(getBit(sortedLabels, dest)).toBe(expectedBit);
    }

    // Check cluster-id ordering: every row in [offsets[c], offsets[c+1]) must
    // map back to a source row whose assignment equals c.
    for (let c = 0; c < K; c++) {
      const lo = offsets[c] as number;
      const hi = offsets[c + 1] as number;
      for (let dest = lo; dest < hi; dest++) {
        const srcI = sortedVectors[dest * D] as number;
        expect(assignments[srcI]).toBe(c);
      }
    }
  });
});
