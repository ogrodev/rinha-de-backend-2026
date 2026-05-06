// Build-time preprocessing pipeline. Stage helpers are exported so the test
// suite can exercise them in isolation; the CLI orchestration in `main` is
// added in Task 12.

// --- Task 7: Streaming gzip + JSON-array parser ---------------------------------

// Reference record schema (locked here for the contest):
//   { "vector": [number x 14], "label": "fraud" | "legit" }
//
// The parser streams the gzipped file and emits one record at a time without
// ever materializing the full decompressed JSON. The caller's `onRecord`
// callback receives a SHARED `Float32Array(14)` reused across calls — copy if
// you need to retain.

export async function parseRefs(
  path: string,
  onRecord: (vec: Float32Array, label: 0 | 1) => void,
): Promise<number> {
  const file = Bun.file(path);
  const stream = file.stream().pipeThrough(new DecompressionStream("gzip"));
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8");
  const vec = new Float32Array(14);

  let buf = "";
  let pos = 0;
  let depth = 0;
  let inString = false;
  let escaping = false;
  let recordStart = -1;
  let sawArrayOpen = false;
  let count = 0;

  const emit = (text: string): void => {
    const obj = JSON.parse(text) as { vector: number[]; label: string };
    const v = obj.vector;
    for (let d = 0; d < 14; d++) vec[d] = v[d] as number;
    onRecord(vec, obj.label === "fraud" ? 1 : 0);
    count++;
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    for (; pos < buf.length; pos++) {
      const c = buf.charCodeAt(pos);

      if (inString) {
        if (escaping) escaping = false;
        else if (c === 92 /* \\ */) escaping = true;
        else if (c === 34 /* " */) inString = false;
        continue;
      }

      if (c === 34 /* " */) { inString = true; continue; }

      if (c === 91 /* [ */) {
        // First `[` opens the outer array; subsequent ones are nested arrays
        // (e.g. inside `vector: [...]`) and count as depth.
        if (!sawArrayOpen) sawArrayOpen = true;
        else depth++;
        continue;
      }
      if (c === 93 /* ] */) {
        if (depth > 0) depth--;
        // depth === 0 here means the outer array is closing; we ignore it.
        continue;
      }
      if (c === 123 /* { */) {
        if (depth === 0 && sawArrayOpen) recordStart = pos;
        depth++;
        continue;
      }
      if (c === 125 /* } */) {
        depth--;
        if (depth === 0 && recordStart >= 0) {
          emit(buf.substring(recordStart, pos + 1));
          recordStart = -1;
        }
        continue;
      }
      // Whitespace and other JSON tokens (numbers, commas, colons, true/false/null)
      // never affect depth/string state — skipped implicitly.
    }

    // Compact `buf`: drop processed prefix to keep memory bounded.
    if (recordStart < 0) {
      buf = buf.substring(pos);
      pos = 0;
    } else if (recordStart > 0) {
      buf = buf.substring(recordStart);
      pos -= recordStart;
      recordStart = 0;
    }
    // If recordStart === 0, buf already starts at the in-progress record.
  }

  // Flush decoder; if the file ended cleanly, no trailing bytes remain.
  buf += decoder.decode();
  return count;
}

// --- Task 8: Quantization scale + int8 buffer build ----------------------------

// Inlined int16 quantizer (mirrors src/index/quantize.ts; kept inline so this
// script can run without depending on runtime sources).
function encodeI16(value: number, scale: number): number {
  const q = Math.round((value * 32767) / scale);
  return q < -32767 ? -32767 : q > 32767 ? 32767 : q;
}

// Compute per-dimension `scale[d] = max(|v|)` over a flat row-major float
// buffer of shape (n, d). Single pass, no allocation beyond the output.
export function buildScale(flat: Float32Array, n: number, d: number): Float32Array {
  const scale = new Float32Array(d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let dim = 0; dim < d; dim++) {
      const v = Math.abs(flat[base + dim] as number);
      if (v > (scale[dim] as number)) scale[dim] = v;
    }
  }
  return scale;
}

// Symmetric int16 quantize a flat (n, d) float buffer using per-dim scales.
// When `scale[dim] === 0` the column is uniformly zero in the input, so we
// emit zeros without dividing.
export function quantizeAll(
  flat: Float32Array,
  scale: Float32Array,
  n: number,
  d: number,
): Int16Array {
  const out = new Int16Array(n * d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let dim = 0; dim < d; dim++) {
      const s = scale[dim] as number;
      out[base + dim] = s === 0 ? 0 : encodeI16(flat[base + dim] as number, s);
    }
  }
  return out;
}

// --- Task 10: Cluster-sort + offsets -------------------------------------------

// Reorder `vectors` (shape n×d, row-major int16) and `labels` (LSB-first packed
// bits) so all rows assigned to cluster c are contiguous, in cluster-id order.
// Returns the reordered buffers plus offsets where cluster c occupies
// [offsets[c], offsets[c+1]). offsets[k] === n.
//
// Counting-sort: O(n) over the data, no comparator.
export function clusterSort(
  vectors: Int16Array,
  labels: Uint8Array,
  assignments: Uint32Array,
  n: number,
  d: number,
  k: number,
): { sortedVectors: Int16Array; sortedLabels: Uint8Array; offsets: Uint32Array } {
  // Pass 1: count per-cluster sizes.
  const offsets = new Uint32Array(k + 1);
  for (let i = 0; i < n; i++) {
    const c = assignments[i] as number;
    offsets[c + 1] = (offsets[c + 1] as number) + 1;
  }
  // Prefix sum.
  for (let c = 1; c <= k; c++) {
    offsets[c] = (offsets[c] as number) + (offsets[c - 1] as number);
  }

  // Pass 2: emit. `cursor[c]` is the next free row inside cluster c.
  const cursor = new Uint32Array(k);
  for (let c = 0; c < k; c++) cursor[c] = offsets[c] as number;

  const sortedVectors = new Int16Array(n * d);
  const sortedLabels = new Uint8Array(Math.ceil(n / 8));

  for (let i = 0; i < n; i++) {
    const c = assignments[i] as number;
    const dest = cursor[c] as number;
    cursor[c] = dest + 1;

    // Copy the row.
    const srcBase = i * d;
    const dstBase = dest * d;
    for (let dim = 0; dim < d; dim++) {
      sortedVectors[dstBase + dim] = vectors[srcBase + dim] as number;
    }
    // Move the label bit.
    const srcBit = ((labels[i >> 3] as number) >> (i & 7)) & 1;
    if (srcBit) sortedLabels[dest >> 3]! |= 1 << (dest & 7);
  }

  return { sortedVectors, sortedLabels, offsets };
}

// --- Task 12: CLI orchestration ------------------------------------------------

import { miniBatchKMeans } from "./kmeans.ts";
import { recallAt5 } from "./recall.ts";

type Args = {
  refs: string;
  norm: string;
  mcc: string;
  out: string;
  k: number;
  iters: number;
  batch: number;
  nprobe: number;
  seed: number;
  recallFloor: number;
  recallSample: number;
};

function parseArgs(argv: string[]): Args {
  const map = new Map<string, string>();
  // Bun.argv[0]/[1] are bun + script path; flags start at [2].
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i] as string;
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      map.set(key, next);
      i++;
    } else {
      map.set(key, "true");
    }
  }
  const required = ["refs", "norm", "mcc", "out"] as const;
  for (const r of required) {
    if (!map.has(r)) {
      console.error(`missing required flag --${r}`);
      process.exit(2);
    }
  }
  const numFlag = (k: string, def: number): number => {
    const v = map.get(k);
    if (v === undefined) return def;
    const n = Number(v);
    if (!Number.isFinite(n)) {
      console.error(`flag --${k} must be a number, got ${JSON.stringify(v)}`);
      process.exit(2);
    }
    return n;
  };
  return {
    refs: map.get("refs") as string,
    norm: map.get("norm") as string,
    mcc: map.get("mcc") as string,
    out: map.get("out") as string,
    k: numFlag("k", 2048),
    iters: numFlag("iters", 25),
    batch: numFlag("batch", 100_000),
    nprobe: numFlag("nprobe", 4),
    seed: numFlag("seed", 42),
    recallFloor: numFlag("recall-floor", 0.99),
    recallSample: numFlag("recall-sample", 1000),
  };
}

const D = 14;

async function loadRefsToFlat(refsPath: string): Promise<{
  flat: Float32Array;
  labels: Uint8Array;
  n: number;
}> {
  let cap = 1024;
  let flat = new Float32Array(cap * D);
  let labels = new Uint8Array(Math.ceil(cap / 8));
  let n = 0;
  await parseRefs(refsPath, (vec, label) => {
    if (n >= cap) {
      const newCap = cap * 2;
      const newFlat = new Float32Array(newCap * D);
      newFlat.set(flat);
      flat = newFlat;
      const newLabels = new Uint8Array(Math.ceil(newCap / 8));
      newLabels.set(labels);
      labels = newLabels;
      cap = newCap;
    }
    const base = n * D;
    for (let d = 0; d < D; d++) flat[base + d] = vec[d] as number;
    if (label === 1) labels[n >> 3]! |= 1 << (n & 7);
    n++;
  });
  // Trim flat to actual size; labels are packed and the trailing bits past n
  // remain zero by construction. Trim the labels byte array exactly.
  const trimmedFlat = flat.length === n * D ? flat : flat.slice(0, n * D);
  const trimmedLabels = labels.length === Math.ceil(n / 8) ? labels : labels.slice(0, Math.ceil(n / 8));
  return { flat: trimmedFlat, labels: trimmedLabels, n };
}

async function copyJson(srcPath: string, destPath: string): Promise<unknown> {
  const value = await Bun.file(srcPath).json();
  await Bun.write(destPath, JSON.stringify(value));
  return value;
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv);

  // Ensure output directory exists.
  const fs = await import("node:fs/promises");
  await fs.mkdir(args.out, { recursive: true });

  // 1) Stream-parse references into a flat float buffer + packed labels.
  console.error(`[preprocess] reading ${args.refs}`);
  const { flat, labels, n } = await loadRefsToFlat(args.refs);
  console.error(`[preprocess] parsed n=${n} records`);
  if (n === 0) {
    console.error("references file produced 0 records");
    process.exit(1);
  }

  // 2) Per-dim scale.
  const scale = buildScale(flat, n, D);

  // 3) int8 quantize.
  const i8 = quantizeAll(flat, scale, n, D);

  // 4) Mini-batch k-means on FLOAT buffer (spec §3.2 step 6).
  console.error(`[preprocess] kmeans k=${args.k} iters=${args.iters} batch=${args.batch}`);
  const k = args.k;
  const { centroids, assignments } = miniBatchKMeans(
    flat,
    n,
    D,
    k,
    args.iters,
    Math.min(args.batch, n),
    args.seed,
  );

  // 5) Cluster-sort the int8 buffer + label bits.
  const { sortedVectors, sortedLabels, offsets } = clusterSort(
    i8,
    labels,
    assignments,
    n,
    D,
    k,
  );

  // 6) Sampled-reference recall@5 gate.
  const recall = recallAt5(
    sortedVectors,
    scale,
    offsets,
    centroids,
    n,
    D,
    k,
    args.nprobe,
    args.recallSample,
    args.seed,
  );
  console.error(`[preprocess] recall@5 = ${recall.toFixed(6)}`);
  await Bun.write(`${args.out}/validation.txt`, `recall@5=${recall}\n`);

  if (recall < args.recallFloor) {
    console.error(`recall=${recall} below floor=${args.recallFloor}`);
    process.exit(1);
  }

  // 7) Copy mcc_risk.json + normalization.json verbatim.
  await copyJson(args.mcc, `${args.out}/mcc_risk.json`);
  await copyJson(args.norm, `${args.out}/normalization.json`);

  // 8) Emit binaries + header.
  const header = {
    n,
    d: D,
    k,
    nprobeDefault: args.nprobe,
    scale: Array.from(scale),
    schemaVersion: 2,
  };
  await Bun.write(`${args.out}/header.json`, JSON.stringify(header));
  await Bun.write(
    `${args.out}/vectors.i16`,
    new Uint8Array(sortedVectors.buffer, sortedVectors.byteOffset, sortedVectors.byteLength),
  );
  await Bun.write(`${args.out}/labels.bits`, sortedLabels);
  await Bun.write(
    `${args.out}/centroids.f32`,
    new Uint8Array(centroids.buffer, centroids.byteOffset, centroids.byteLength),
  );
  await Bun.write(
    `${args.out}/offsets.u32`,
    new Uint8Array(offsets.buffer, offsets.byteOffset, offsets.byteLength),
  );
  console.error("[preprocess] done");
}

// Run only when invoked as the main entry, not when imported by tests.
if (import.meta.main) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
