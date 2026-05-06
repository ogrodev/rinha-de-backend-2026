import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadIndex } from "../src/index/load.ts";
import { search, makeScratch } from "../src/index/search.ts";
import type { Index } from "../src/index/types.ts";
import {
  buildScale,
  quantizeAll,
  clusterSort,
} from "../scripts/preprocess.ts";
import { miniBatchKMeans, lcg } from "../scripts/kmeans.ts";
import { topKInit, topKConsider } from "../src/util/topk.ts";

const D = 14;

function gauss(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function synthDataset(n: number, seed: number): { flat: Float32Array; labels: Uint8Array } {
  const flat = new Float32Array(n * D);
  const labels = new Uint8Array(Math.ceil(n / 8));
  const rng = lcg(seed);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) {
      flat[i * D + d] = gauss(rng);
    }
    if (i % 9 === 0) labels[i >> 3]! |= 1 << (i & 7);
  }
  return { flat, labels };
}

async function buildAndLoadIndex(n: number, K: number, seed: number): Promise<Index> {
  const { flat, labels } = synthDataset(n, seed);
  const scale = buildScale(flat, n, D);
  const i8 = quantizeAll(flat, scale, n, D);
  const { centroids, assignments } = miniBatchKMeans(flat, n, D, K, 10, 200, seed);
  const { sortedVectors, sortedLabels, offsets } = clusterSort(
    i8,
    labels,
    assignments,
    n,
    D,
    K,
  );

  const dir = mkdtempSync(join(tmpdir(), "rinha-parity-"));
  const header = {
    n,
    d: 14,
    k: K,
    nprobeDefault: 4,
    scale: Array.from(scale),
    schemaVersion: 3,
  };
  await Bun.write(join(dir, "header.json"), JSON.stringify(header));
  await Bun.write(join(dir, "vectors.i16"), sortedVectors);
  await Bun.write(join(dir, "labels.bits"), sortedLabels);
  await Bun.write(
    join(dir, "centroids.f32"),
    new Uint8Array(centroids.buffer, centroids.byteOffset, centroids.byteLength),
  );
  const radii = new Float32Array(K).fill(1e30);
  await Bun.write(
    join(dir, "radii.f32"),
    new Uint8Array(radii.buffer, radii.byteOffset, radii.byteLength),
  );
  await Bun.write(
    join(dir, "offsets.u32"),
    new Uint8Array(offsets.buffer, offsets.byteOffset, offsets.byteLength),
  );
  await Bun.write(join(dir, "mcc_risk.json"), "{}");
  await Bun.write(
    join(dir, "normalization.json"),
    JSON.stringify({
      max_amount: 10000, max_installments: 12, amount_vs_avg_ratio: 10,
      max_minutes: 1440, max_km: 1000, max_tx_count_24h: 20,
      max_merchant_avg_amount: 10000,
    }),
  );

  return await loadIndex(dir);
}

// Brute-force top-5 over the same int8 vectors using the SAME insertion-sort
// helper used by `search`. This guarantees identical tie-breaking, so the
// approx (with nprobe=k) and exact methods produce bitwise-equal index sets.
function bruteForceTop5(idx: Index, query: Float32Array): Uint32Array {
  const dist = new Float32Array(5);
  const ids = new Uint32Array(5);
  topKInit(dist, ids);
  const D = idx.d;
  for (let i = 0; i < idx.n; i++) {
    const base = i * D;
    let s = 0;
    for (let dim = 0; dim < D; dim++) {
      const v = (idx.vectors[base + dim] as number) * (idx.decodeFactor[dim] as number);
      const diff = (query[dim] as number) - v;
      s += diff * diff;
    }
    topKConsider(dist, ids, s, i);
  }
  return ids;
}

describe("IVF search parity", () => {
  test("nprobe=k produces identical top-5 index set as brute force (200 random queries)", async () => {
    const n = 10_000;
    const K = 32;
    const idx = await buildAndLoadIndex(n, K, 99);
    // Force nprobe == k by mutating the loaded structure (test-only).
    (idx as any).nprobe = K;
    const scratch = makeScratch(K, K);

    const rng = lcg(101);
    for (let q = 0; q < 200; q++) {
      // Sample a query vector by dequantizing a random reference row.
      let qi = Math.floor(rng() * n);
      if (qi >= n) qi = n - 1;
      const query = new Float32Array(D);
      for (let dim = 0; dim < D; dim++) {
        query[dim] = (idx.vectors[qi * D + dim] as number) * (idx.decodeFactor[dim] as number);
      }

      // Run both methods.
      search(idx, query, scratch);
      const approx = Array.from(scratch.top5Idx).slice().sort((a, b) => a - b);
      const exact = Array.from(bruteForceTop5(idx, query)).slice().sort((a, b) => a - b);
      expect(approx).toEqual(exact);
    }
  }, 60_000);

  test("zero-allocation across 50000 queries with reused scratch", async () => {
    const n = 2_000;
    const K = 16;
    const idx = await buildAndLoadIndex(n, K, 7);
    const scratch = makeScratch(K, idx.nprobe);
    const query = new Float32Array(D);
    // Use a deterministic reference row as query.
    for (let dim = 0; dim < D; dim++) {
      query[dim] = (idx.vectors[0 + dim] as number) * (idx.decodeFactor[dim] as number);
    }

    if (typeof Bun !== "undefined" && (Bun as any).gc) (Bun as any).gc(true);
    const before = process.memoryUsage().heapUsed;
    for (let q = 0; q < 50_000; q++) {
      search(idx, query, scratch);
    }
    const after = process.memoryUsage().heapUsed;
    expect(after - before).toBeLessThan(5 * 1024 * 1024);
  }, 60_000);
});
