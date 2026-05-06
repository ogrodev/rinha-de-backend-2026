import { describe, expect, test } from "bun:test";
import {
  lcg,
  nearestCentroidIndex,
  kmeansPlusPlusSeed,
  miniBatchUpdate,
  miniBatchKMeans,
} from "../scripts/kmeans.ts";

// --- lcg ---------------------------------------------------------------------

describe("lcg", () => {
  test("yields floats in [0, 1)", () => {
    const r = lcg(42);
    for (let i = 0; i < 5; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  test("identical seeds produce identical sequences", () => {
    const a = lcg(42);
    const b = lcg(42);
    for (let i = 0; i < 10; i++) {
      expect(a()).toBe(b());
    }
  });

  test("different seeds diverge at index 0", () => {
    const a = lcg(42);
    const b = lcg(43);
    expect(a()).not.toBe(b());
  });
});

// --- nearestCentroidIndex ---------------------------------------------------

describe("nearestCentroidIndex", () => {
  test("3 centroids, 2 dims — picks the one closest in L2", () => {
    const centroids = new Float32Array([
      0, 0,   // c=0
      10, 0,  // c=1
      0, 10,  // c=2
    ]);
    const point = new Float32Array([1, 9]); // closer to c=2
    expect(nearestCentroidIndex(point, 0, centroids, 3, 2)).toBe(2);

    const point2 = new Float32Array([9, 1]); // closer to c=1
    expect(nearestCentroidIndex(point2, 0, centroids, 3, 2)).toBe(1);

    const point3 = new Float32Array([0.1, 0.1]); // closer to c=0
    expect(nearestCentroidIndex(point3, 0, centroids, 3, 2)).toBe(0);
  });
});

// --- kmeans++ seeding -------------------------------------------------------

function makeFourGaussianClusters(seed: number): { flat: Float32Array; n: number; d: number; trueLabel: Uint8Array } {
  const d = 14;
  const perCluster = 250;
  const n = 4 * perCluster;
  const flat = new Float32Array(n * d);
  const trueLabel = new Uint8Array(n);
  // Cluster centers: place far apart in the first 2 dims, zero elsewhere.
  const centers: number[][] = [
    [0, 0], [100, 0], [0, 100], [100, 100],
  ];
  const rng = lcg(seed);
  // Box-Muller for Gaussians.
  function normal(): number {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  for (let c = 0; c < 4; c++) {
    for (let i = 0; i < perCluster; i++) {
      const row = c * perCluster + i;
      const base = row * d;
      flat[base + 0] = (centers[c]![0] as number) + normal() * 1.0;
      flat[base + 1] = (centers[c]![1] as number) + normal() * 1.0;
      // remaining dims small noise
      for (let dim = 2; dim < d; dim++) flat[base + dim] = normal() * 0.1;
      trueLabel[row] = c;
    }
  }
  return { flat, n, d, trueLabel };
}

describe("kmeansPlusPlusSeed", () => {
  test("on 4 well-separated clusters, picks 4 distinct generating clusters", () => {
    const { flat, n, d, trueLabel } = makeFourGaussianClusters(42);
    const { chosenIndices } = kmeansPlusPlusSeed(flat, n, d, 4, lcg(42));
    const distinctClusters = new Set<number>();
    for (let i = 0; i < 4; i++) {
      distinctClusters.add(trueLabel[chosenIndices[i] as number] as number);
    }
    expect(distinctClusters.size).toBe(4);
  });
});

// --- miniBatchUpdate --------------------------------------------------------

describe("miniBatchUpdate", () => {
  test("centroids move toward sampled points' cluster mean", () => {
    // 4 well-spaced 2D centers, 10 points per cluster, sample 50 random points.
    const d = 2;
    const k = 4;
    const perCluster = 10;
    const n = perCluster * k;
    const flat = new Float32Array(n * d);
    const centers: number[][] = [[0, 0], [10, 0], [0, 10], [10, 10]];
    for (let c = 0; c < k; c++) {
      for (let i = 0; i < perCluster; i++) {
        const row = c * perCluster + i;
        flat[row * d + 0] = (centers[c]![0] as number) + (i - perCluster / 2) * 0.1;
        flat[row * d + 1] = (centers[c]![1] as number) + (i - perCluster / 2) * 0.1;
      }
    }
    // Initialize centroids slightly off true centers.
    const centroids = new Float32Array(k * d);
    for (let c = 0; c < k; c++) {
      centroids[c * d + 0] = (centers[c]![0] as number) + 1.0;
      centroids[c * d + 1] = (centers[c]![1] as number) + 1.0;
    }
    const counts = new Float32Array(k);
    const sample = new Uint32Array(50);
    for (let s = 0; s < 50; s++) sample[s] = s % n;

    // Compute pre/post squared distance from each centroid to its true center.
    function distToTrueCenter(): number[] {
      const out: number[] = [];
      for (let c = 0; c < k; c++) {
        const cx = centroids[c * d + 0] as number;
        const cy = centroids[c * d + 1] as number;
        const tx = centers[c]![0] as number;
        const ty = centers[c]![1] as number;
        out.push((cx - tx) * (cx - tx) + (cy - ty) * (cy - ty));
      }
      return out;
    }

    const before = distToTrueCenter();
    miniBatchUpdate(centroids, counts, flat, sample, d, k);
    const after = distToTrueCenter();

    for (let c = 0; c < k; c++) {
      expect(after[c]).toBeLessThan(before[c] as number);
    }
  });
});

// --- miniBatchKMeans (integration) -----------------------------------------

describe("miniBatchKMeans", () => {
  test("on 4-cluster Gaussian synthetic data, ≥99% points land in their generating cluster", () => {
    const { flat, n, d, trueLabel } = makeFourGaussianClusters(7);
    const { assignments } = miniBatchKMeans(flat, n, d, 4, 20, 200, 42);

    // Confusion matrix: rows = predicted cluster, cols = true cluster.
    const conf: number[][] = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];
    for (let i = 0; i < n; i++) {
      conf[assignments[i] as number]![trueLabel[i] as number]!++;
    }
    // For each predicted cluster, choose the true cluster with the most hits.
    // Hungarian-lite: pick max per row, but ensure unique mapping.
    const used = new Set<number>();
    let correct = 0;
    // Sort predicted-cluster rows by their best column count, descending,
    // and greedily assign.
    const rowOrder = [0, 1, 2, 3].sort((a, b) => {
      const bMax = Math.max(...(conf[b] as number[]));
      const aMax = Math.max(...(conf[a] as number[]));
      return bMax - aMax;
    });
    for (const r of rowOrder) {
      let best = -1;
      let bestCount = -1;
      for (let c = 0; c < 4; c++) {
        if (used.has(c)) continue;
        const v = conf[r]![c] as number;
        if (v > bestCount) {
          bestCount = v;
          best = c;
        }
      }
      if (best >= 0) {
        used.add(best);
        correct += bestCount;
      }
    }
    expect(correct / n).toBeGreaterThanOrEqual(0.99);
  });
});
