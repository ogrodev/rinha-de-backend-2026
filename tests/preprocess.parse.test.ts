import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseRefs } from "../scripts/preprocess.ts";

describe("parseRefs (streaming gzip + JSON-array parser)", () => {
  test("emits 5 records in order with vectors and labels", async () => {
    const records = [
      { vector: new Array(14).fill(0).map((_, i) => i / 10), label: "legit" },
      { vector: new Array(14).fill(0).map((_, i) => 1 + i / 10), label: "fraud" },
      { vector: new Array(14).fill(0).map((_, i) => 2 + i / 10), label: "legit" },
      { vector: new Array(14).fill(0).map((_, i) => 3 + i / 10), label: "fraud" },
      { vector: new Array(14).fill(0).map((_, i) => 4 + i / 10), label: "legit" },
    ];
    const json = JSON.stringify(records);
    const gz = Bun.gzipSync(new TextEncoder().encode(json));
    const dir = mkdtempSync(join(tmpdir(), "rinha-parse-"));
    const file = join(dir, "refs.json.gz");
    writeFileSync(file, gz);

    const seen: Array<{ vec: number[]; label: 0 | 1 }> = [];
    const count = await parseRefs(file, (vec, label) => {
      // Caller copies because the parser reuses the buffer.
      seen.push({ vec: Array.from(vec), label });
    });

    expect(count).toBe(5);
    expect(seen).toHaveLength(5);
    for (let i = 0; i < 5; i++) {
      const expectedVec = records[i]!.vector;
      const expectedLabel: 0 | 1 = records[i]!.label === "fraud" ? 1 : 0;
      const actual = seen[i]!;
      expect(actual.label).toBe(expectedLabel);
      for (let d = 0; d < 14; d++) {
        expect(actual.vec[d]).toBeCloseTo(expectedVec[d] as number, 6);
      }
    }
  });

  test("peak heap stays under 50 MB while parsing 10000 records", async () => {
    const records: Array<{ vector: number[]; label: string }> = [];
    for (let i = 0; i < 10_000; i++) {
      const vector: number[] = new Array(14);
      for (let d = 0; d < 14; d++) vector[d] = Math.sin(i * 14 + d);
      records.push({ vector, label: i % 7 === 0 ? "fraud" : "legit" });
    }
    const json = JSON.stringify(records);
    const gz = Bun.gzipSync(new TextEncoder().encode(json));
    const dir = mkdtempSync(join(tmpdir(), "rinha-parse-"));
    const file = join(dir, "refs.json.gz");
    writeFileSync(file, gz);

    if (typeof Bun !== "undefined" && (Bun as any).gc) (Bun as any).gc(true);
    const baseline = process.memoryUsage().heapUsed;
    let peak = baseline;
    let frauds = 0;

    const count = await parseRefs(file, (_vec, label) => {
      if (label === 1) frauds++;
      // Sample heap usage on first and last record.
      const used = process.memoryUsage().heapUsed;
      if (used > peak) peak = used;
    });

    expect(count).toBe(10_000);
    // Roughly 10000/7 = 1428 frauds.
    expect(frauds).toBeGreaterThan(1400);
    expect(frauds).toBeLessThan(1500);
    expect(peak - baseline).toBeLessThan(50 * 1024 * 1024);
  });
});
