// Build a synthetic ./resources/ directory for local docker smoke-tests.
// NOT for production — replace with the real contest dataset before running
// the official acceptance pipeline (Tasks 24/25).

import { writeFileSync, mkdirSync } from "node:fs";

const D = 14;
const N = Number(process.argv[2] ?? 5000);

const recs: Array<{ vector: number[]; label: string }> = [];
let s = 0xCAFEBABE >>> 0;
const rng = () => {
  s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
  return s / 0x100000000;
};
function gauss(): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
const k = 16;
const centers: number[][] = [];
for (let c = 0; c < k; c++) {
  const center: number[] = new Array(D).fill(0);
  center[0] = (c % 4) * 50;
  center[1] = Math.floor(c / 4) * 50;
  centers.push(center);
}
for (let i = 0; i < N; i++) {
  const c = i % k;
  const v: number[] = new Array(D);
  for (let d = 0; d < D; d++) v[d] = (centers[c]![d] as number) + gauss() * 0.5;
  recs.push({ vector: v, label: i % 9 === 0 ? "fraud" : "legit" });
}

mkdirSync("resources", { recursive: true });
const json = JSON.stringify(recs);
writeFileSync("resources/references.json.gz", Bun.gzipSync(new TextEncoder().encode(json)));
writeFileSync(
  "resources/normalization.json",
  JSON.stringify({
    max_amount: 10000,
    max_installments: 12,
    amount_vs_avg_ratio: 10,
    max_minutes: 1440,
    max_km: 1000,
    max_tx_count_24h: 20,
    max_merchant_avg_amount: 10000,
  }),
);
writeFileSync(
  "resources/mcc_risk.json",
  JSON.stringify({ "5411": 0.15, "7802": 0.75, "5812": 0.4 }),
);
console.log(`wrote synthetic resources/ with N=${N}`);
