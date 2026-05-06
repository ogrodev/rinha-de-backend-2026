import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadIndex } from "../src/index/load.ts";

const D = 14;

async function buildSyntheticIndex(opts?: {
  truncateVectors?: boolean;
  badSchema?: boolean;
  missingHeader?: boolean;
}): Promise<string> {
  const N = 100;
  const K = 8;
  const dir = mkdtempSync(join(tmpdir(), "rinha-load-"));
  mkdirSync(dir, { recursive: true });

  const scale = new Float32Array(D);
  for (let d = 0; d < D; d++) scale[d] = 1 + d * 0.1;

  const vectors = new Int16Array(N * D);
  for (let i = 0; i < N * D; i++) vectors[i] = ((i * 7) % 254) - 127;
  const labels = new Uint8Array(Math.ceil(N / 8));
  for (let i = 0; i < N; i++) {
    if (i % 3 === 0) labels[i >> 3]! |= 1 << (i & 7);
  }
  const centroids = new Float32Array(K * D);
  for (let i = 0; i < centroids.length; i++) centroids[i] = i * 0.01;
  const offsets = new Uint32Array(K + 1);
  for (let c = 0; c <= K; c++) offsets[c] = Math.floor((c * N) / K);

  if (!opts?.missingHeader) {
    const header = {
      n: N,
      d: 14,
      k: K,
      nprobeDefault: 4,
      scale: Array.from(scale),
      schemaVersion: opts?.badSchema ? 99 : 2,
    };
    await Bun.write(join(dir, "header.json"), JSON.stringify(header));
  }

  if (opts?.truncateVectors) {
    await Bun.write(join(dir, "vectors.i16"), vectors.subarray(0, vectors.length - D));
  } else {
    await Bun.write(join(dir, "vectors.i16"), vectors);
  }
  await Bun.write(join(dir, "labels.bits"), labels);
  await Bun.write(
    join(dir, "centroids.f32"),
    new Uint8Array(centroids.buffer, centroids.byteOffset, centroids.byteLength),
  );
  await Bun.write(
    join(dir, "offsets.u32"),
    new Uint8Array(offsets.buffer, offsets.byteOffset, offsets.byteLength),
  );

  await Bun.write(
    join(dir, "mcc_risk.json"),
    JSON.stringify({ "5411": 0.15, "7802": 0.75 }),
  );
  await Bun.write(
    join(dir, "normalization.json"),
    JSON.stringify({
      max_amount: 10000, max_installments: 12, amount_vs_avg_ratio: 10,
      max_minutes: 1440, max_km: 1000, max_tx_count_24h: 20,
      max_merchant_avg_amount: 10000,
    }),
  );

  return dir;
}

describe("loadIndex", () => {
  test("loads valid index, sets decodeFactor, exposes maps and norm", async () => {
    const dir = await buildSyntheticIndex();
    const idx = await loadIndex(dir);

    expect(idx.n).toBe(100);
    expect(idx.d).toBe(14);
    expect(idx.k).toBe(8);
    expect(idx.nprobe).toBe(4);
    expect(idx.scale.length).toBe(14);
    expect(idx.decodeFactor.length).toBe(14);
    for (let d = 0; d < 14; d++) {
      expect(idx.decodeFactor[d]).toBeCloseTo((idx.scale[d] as number) / 32767, 6);
    }
    expect(idx.vectors.length).toBe(100 * 14);
    expect(idx.labels.length).toBe(Math.ceil(100 / 8));
    expect(idx.centroids.length).toBe(8 * 14);
    expect(idx.offsets.length).toBe(8 + 1);
    expect(idx.offsets[8]).toBe(100);
    expect(idx.mccRisk.get("5411")).toBe(0.15);
    expect(idx.norm.max_amount).toBe(10000);
  });

  test("throws on missing header.json", async () => {
    const dir = await buildSyntheticIndex({ missingHeader: true });
    await expect(loadIndex(dir)).rejects.toThrow(/header\.json/);
  });

  test("throws on schema version mismatch", async () => {
    const dir = await buildSyntheticIndex({ badSchema: true });
    await expect(loadIndex(dir)).rejects.toThrow(/schemaVersion|header\.json/);
  });

  test("throws on truncated vectors.i16", async () => {
    const dir = await buildSyntheticIndex({ truncateVectors: true });
    await expect(loadIndex(dir)).rejects.toThrow(/vectors\.i16/);
  });
});
