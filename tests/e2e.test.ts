import { describe, expect, test, beforeAll, afterAll } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildScale,
  quantizeAll,
  clusterSort,
} from "../scripts/preprocess.ts";
import { miniBatchKMeans, lcg } from "../scripts/kmeans.ts";

const D = 14;

async function buildSyntheticIndexDir(): Promise<string> {
  const n = 1024;
  const K = 32;
  const flat = new Float32Array(n * D);
  const labels = new Uint8Array(Math.ceil(n / 8));
  const rng = lcg(2024);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) flat[i * D + d] = (rng() - 0.5) * 2;
    if (i % 5 === 0) labels[i >> 3]! |= 1 << (i & 7); // ~20% fraud
  }
  const scale = buildScale(flat, n, D);
  const i8 = quantizeAll(flat, scale, n, D);
  const { centroids, assignments } = miniBatchKMeans(flat, n, D, K, 5, 200, 7);
  const { sortedVectors, sortedLabels, offsets } = clusterSort(
    i8,
    labels,
    assignments,
    n,
    D,
    K,
  );
  const dir = mkdtempSync(join(tmpdir(), "rinha-e2e-"));
  await Bun.write(
    join(dir, "header.json"),
    JSON.stringify({ n, d: 14, k: K, nprobeDefault: 4, scale: Array.from(scale), schemaVersion: 3 }),
  );
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
  await Bun.write(join(dir, "mcc_risk.json"), JSON.stringify({ "5411": 0.15 }));
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

const PORT = 33800 + Math.floor(Math.random() * 1000);
let proc: ReturnType<typeof Bun.spawn> | null = null;

beforeAll(async () => {
  const dataDir = await buildSyntheticIndexDir();
  proc = Bun.spawn({
    cmd: ["bun", "src/server.ts"],
    env: { ...process.env, DATA_DIR: dataDir, PORT: String(PORT) },
    cwd: process.cwd(),
    stdout: "pipe",
    stderr: "pipe",
  });
  // Poll /ready up to 10s.
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/ready`);
      if (r.status === 200) return;
    } catch {
      // not yet listening
    }
    await Bun.sleep(50);
  }
  throw new Error("server did not become ready within 10s");
}, 30_000);

afterAll(async () => {
  if (proc) {
    proc.kill();
    await proc.exited;
  }
});

describe("e2e: real Bun.serve round-trip", () => {
  test("/ready returns 200 {}", async () => {
    const r = await fetch(`http://127.0.0.1:${PORT}/ready`);
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toBe("application/json");
    expect(await r.text()).toBe("{}");
  });

  test("POST /fraud-score returns scored body in expected shape", async () => {
    const payload = {
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
      customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-002", mcc: "5411", avg_amount: 60 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
    };
    const r = await fetch(`http://127.0.0.1:${PORT}/fraud-score`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toBe("application/json");
    const body = (await r.json()) as { approved: boolean; fraud_score: number };
    expect(typeof body.approved).toBe("boolean");
    expect([0, 0.2, 0.4, 0.6, 0.8, 1.0]).toContain(body.fraud_score);
    expect(body.approved).toBe(body.fraud_score < 0.6);
  });

  test("malformed JSON returns 400 invalid_json", async () => {
    const r = await fetch(`http://127.0.0.1:${PORT}/fraud-score`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{ broken",
    });
    expect(r.status).toBe(400);
    expect(await r.text()).toBe('{"error":"invalid_json"}');
  });

  test("unknown route returns 404", async () => {
    const r = await fetch(`http://127.0.0.1:${PORT}/nope`);
    expect(r.status).toBe(404);
  });
});
