import { describe, expect, test } from "bun:test";
import { recallAt5 } from "../scripts/recall.ts";
import {
  buildScale,
  quantizeAll,
  clusterSort,
} from "../scripts/preprocess.ts";
import { miniBatchKMeans, lcg } from "../scripts/kmeans.ts";

const D = 14;

function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function synth10kClusters(seed: number): { flat: Float32Array; n: number } {
  const k = 16; // 16 well-spaced cluster centers in 14-d
  const perCluster = 625;
  const n = k * perCluster;
  const flat = new Float32Array(n * D);
  const rng = lcg(seed);
  // Generate cluster centers far apart in the first 4 dims.
  const centers: number[][] = [];
  for (let c = 0; c < k; c++) {
    const center: number[] = new Array(D).fill(0);
    center[0] = (c % 4) * 50;
    center[1] = Math.floor(c / 4) * 50;
    center[2] = ((c * 13) % 5) * 30;
    center[3] = ((c * 7) % 3) * 30;
    centers.push(center);
  }
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < perCluster; i++) {
      const row = c * perCluster + i;
      const base = row * D;
      for (let d = 0; d < D; d++) {
        flat[base + d] = (centers[c]![d] as number) + gaussian(rng) * 0.5;
      }
    }
  }
  return { flat, n };
}

describe("recallAt5", () => {
  test("≥ 0.99 on synthetic well-clustered data with nprobe=4", () => {
    const { flat, n } = synth10kClusters(11);
    const K = 64;

    // Standard preprocessing pipeline: scale → quantize → cluster → sort.
    const scale = buildScale(flat, n, D);
    const i8 = quantizeAll(flat, scale, n, D);
    const labels = new Uint8Array(Math.ceil(n / 8)); // unused for recall
    const { centroids, assignments } = miniBatchKMeans(flat, n, D, K, 20, 500, 7);
    const { sortedVectors, offsets } = clusterSort(
      i8,
      labels,
      assignments,
      n,
      D,
      K,
    );

    const recall = recallAt5(
      sortedVectors,
      scale,
      offsets,
      centroids,
      n,
      D,
      K,
      4,
      200,
      7,
    );
    expect(recall).toBeGreaterThanOrEqual(0.99);
  });
});
