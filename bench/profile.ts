// Inner-loop latency profiler. Loads the preprocessed index from DATA_DIR,
// samples in-distribution queries (random reference rows dequantized on the
// fly), and prints p50/p95/p99/p99.9 over `BENCH_N` searches. Defaults to
// 100,000 iterations.
//
// Usage: DATA_DIR=./data BENCH_N=100000 bun bench/profile.ts

import { loadIndex } from "../src/index/load.ts";
import { makeScratch, search } from "../src/index/search.ts";

const DATA_DIR = process.env.DATA_DIR ?? "./data";
const N = Number(process.env.BENCH_N ?? 100_000);

function pct(sortedMs: Float64Array, p: number): number {
  const idx = Math.min(sortedMs.length - 1, Math.floor((p / 100) * sortedMs.length));
  return sortedMs[idx] as number;
}

async function main(): Promise<void> {
  const t0 = Bun.nanoseconds();
  const idx = await loadIndex(DATA_DIR);
  console.error(
    `[bench] loaded n=${idx.n} k=${idx.k} d=${idx.d} nprobe=${idx.nprobe} in ${
      ((Bun.nanoseconds() - t0) / 1e6).toFixed(1)
    } ms`,
  );

  const scratch = makeScratch(idx.k, idx.nprobe);
  const query = new Float32Array(idx.d);
  const samples = new Float64Array(N);

  // Warm the JIT before timing.
  for (let w = 0; w < 1000; w++) {
    const qi = w % idx.n;
    for (let dim = 0; dim < idx.d; dim++) {
      query[dim] = (idx.vectors[qi * idx.d + dim] as number) * (idx.decodeFactor[dim] as number);
    }
    search(idx, query, scratch);
  }

  // Deterministic sampling — Math.random replaced by a tiny LCG for
  // reproducible bench runs.
  let s = 0xDEADBEEF >>> 0;
  for (let q = 0; q < N; q++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const qi = s % idx.n;
    // Materialize the query from row qi.
    for (let dim = 0; dim < idx.d; dim++) {
      query[dim] = (idx.vectors[qi * idx.d + dim] as number) * (idx.decodeFactor[dim] as number);
    }
    const t = Bun.nanoseconds();
    search(idx, query, scratch);
    samples[q] = (Bun.nanoseconds() - t) / 1e6;
  }

  // Sort and report percentiles.
  samples.sort();
  console.log(
    [
      `n=${N}`,
      `p50=${pct(samples, 50).toFixed(3)}ms`,
      `p95=${pct(samples, 95).toFixed(3)}ms`,
      `p99=${pct(samples, 99).toFixed(3)}ms`,
      `p99.9=${pct(samples, 99.9).toFixed(3)}ms`,
      `max=${(samples[samples.length - 1] as number).toFixed(3)}ms`,
    ].join(" "),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
