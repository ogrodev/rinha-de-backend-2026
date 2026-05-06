// Compute sampled-reference recall@5 across a sweep of nprobe values, using
// the binaries already built in DATA_DIR (default ./data-real). Cheap because
// it reuses the existing centroids/offsets/vectors and doesn't re-run kmeans.

import { recallAt5 } from "./recall.ts";

const dir = process.argv[2] ?? "data-real";
const sample = Number(process.argv[3] ?? "1000");

const header = await Bun.file(`${dir}/header.json`).json() as {
  n: number; d: number; k: number; nprobeDefault: number; scale: number[];
};
const n = header.n, k = header.k, D = header.d;

const vectorsBuf = await Bun.file(`${dir}/vectors.i16`).arrayBuffer();
const centroidsBuf = await Bun.file(`${dir}/centroids.f32`).arrayBuffer();
const offsetsBuf = await Bun.file(`${dir}/offsets.u32`).arrayBuffer();

const vectors = new Int16Array(vectorsBuf);
const centroids = new Float32Array(centroidsBuf);
const offsets = new Uint32Array(offsetsBuf);
const scale = new Float32Array(header.scale);

console.log(`n=${n} k=${k} D=${D} sample=${sample}`);
for (const nprobe of [4, 8, 16, 32, 64, 128, 256]) {
  if (nprobe > k) continue;
  const t0 = Bun.nanoseconds();
  const r = recallAt5(vectors, scale, offsets, centroids, n, D, k, nprobe, sample, 7);
  const ms = (Bun.nanoseconds() - t0) / 1e6;
  console.log(`  nprobe=${String(nprobe).padStart(3)}  recall@5=${r.toFixed(5)}  (${ms.toFixed(0)}ms)`);
}
