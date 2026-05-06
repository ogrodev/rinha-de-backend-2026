import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function makeWellClusteredRefs(n: number, k: number): Buffer {
  const D = 14;
  const records: Array<{ vector: number[]; label: string }> = [];
  // k cluster centers, deterministic.
  const centers: number[][] = [];
  for (let c = 0; c < k; c++) {
    const center: number[] = new Array(D).fill(0);
    center[0] = (c % 8) * 50;
    center[1] = Math.floor(c / 8) * 50;
    center[2] = ((c * 13) % 5) * 30;
    centers.push(center);
  }
  let s = 12345 >>> 0;
  const lcg = (): number => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  function gauss(): number {
    const u1 = Math.max(lcg(), 1e-12);
    const u2 = lcg();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  for (let i = 0; i < n; i++) {
    const c = i % k;
    const vector: number[] = new Array(D);
    for (let d = 0; d < D; d++) {
      vector[d] = (centers[c]![d] as number) + gauss() * 0.5;
    }
    records.push({ vector, label: i % 17 === 0 ? "fraud" : "legit" });
  }
  const json = JSON.stringify(records);
  return Buffer.from(Bun.gzipSync(new TextEncoder().encode(json)));
}

async function setupTempInputs(): Promise<{ dir: string; refsPath: string; normPath: string; mccPath: string }> {
  const dir = mkdtempSync(join(tmpdir(), "rinha-pp-"));
  const refsPath = join(dir, "refs.json.gz");
  writeFileSync(refsPath, makeWellClusteredRefs(10_000, 64));

  const normPath = join(dir, "normalization.json");
  writeFileSync(normPath, JSON.stringify({
    max_amount: 10000, max_installments: 12, amount_vs_avg_ratio: 10,
    max_minutes: 1440, max_km: 1000, max_tx_count_24h: 20,
    max_merchant_avg_amount: 10000,
  }));

  const mccPath = join(dir, "mcc_risk.json");
  writeFileSync(mccPath, JSON.stringify({ "5411": 0.15, "7802": 0.75 }));

  return { dir, refsPath, normPath, mccPath };
}

async function runPreprocess(args: string[]): Promise<{ exitCode: number; stderr: string }> {
  const proc = Bun.spawn({
    cmd: ["bun", "scripts/preprocess.ts", ...args],
    cwd: process.cwd(),
    stdout: "pipe",
    stderr: "pipe",
  });
  const stderrText = await new Response(proc.stderr).text();
  await proc.exited;
  return { exitCode: proc.exitCode ?? -1, stderr: stderrText };
}

describe("preprocess CLI", () => {
  test("happy path: produces all expected files with correct header", async () => {
    const { dir, refsPath, normPath, mccPath } = await setupTempInputs();
    const outDir = join(dir, "out");
    const result = await runPreprocess([
      "--refs", refsPath,
      "--norm", normPath,
      "--mcc", mccPath,
      "--out", outDir,
      "--k", "64",
      "--iters", "5",
      "--batch", "1000",
      "--recall-sample", "100",
    ]);
    expect(result.exitCode).toBe(0);

    for (const fname of [
      "header.json",
      "vectors.i16",
      "labels.bits",
      "centroids.f32",
      "radii.f32",
      "offsets.u32",
      "validation.txt",
      "mcc_risk.json",
      "normalization.json",
    ]) {
      expect(existsSync(join(outDir, fname))).toBe(true);
    }

    const header = await Bun.file(join(outDir, "header.json")).json();
    expect(header.n).toBe(10_000);
    expect(header.d).toBe(14);
    expect(header.k).toBe(64);
    expect(header.nprobeDefault).toBe(4);
    expect(header.schemaVersion).toBe(3);
    expect(header.scale.length).toBe(14);

    // Binary sizes match header.
    expect(Bun.file(join(outDir, "vectors.i16")).size).toBe(10_000 * 14 * 2);
    expect(Bun.file(join(outDir, "labels.bits")).size).toBe(Math.ceil(10_000 / 8));
    expect(Bun.file(join(outDir, "centroids.f32")).size).toBe(64 * 14 * 4);
    expect(Bun.file(join(outDir, "offsets.u32")).size).toBe((64 + 1) * 4);
    expect(Bun.file(join(outDir, "radii.f32")).size).toBe(64 * 4);
  }, 60_000);

  test("recall floor 1.01 trips the gate (deterministic failure)", async () => {
    const { dir, refsPath, normPath, mccPath } = await setupTempInputs();
    const outDir = join(dir, "out-fail");
    const result = await runPreprocess([
      "--refs", refsPath,
      "--norm", normPath,
      "--mcc", mccPath,
      "--out", outDir,
      "--k", "64",
      "--iters", "5",
      "--batch", "1000",
      "--recall-sample", "100",
      "--recall-floor", "1.01",
    ]);
    expect(result.exitCode).not.toBe(0);
    expect(result.stderr.toLowerCase()).toContain("recall");
  }, 60_000);
});
