// Cold-start index loader (spec §4.1).
//
// Reads the five binaries plus mcc_risk.json and normalization.json, validates
// header invariants and binary sizes, and constructs typed-array views with
// no copy. Throws an Error whose message names the offending file on any
// inconsistency so docker logs make the cause obvious.

import { join } from "node:path";
import type { Index, NormConsts } from "./types.ts";
import { decodeFactor as decodeFactorOf } from "./quantize.ts";

type Header = {
  readonly n: number;
  readonly d: number;
  readonly k: number;
  readonly nprobeDefault: number;
  readonly scale: number[];
  readonly schemaVersion: number;
};

const SCHEMA_VERSION = 2;
const D = 14;

async function readJson<T>(path: string, label: string): Promise<T> {
  const f = Bun.file(path);
  if (!(await f.exists())) {
    throw new Error(`missing ${label}: ${path}`);
  }
  try {
    return (await f.json()) as T;
  } catch (err) {
    throw new Error(`failed to parse ${label} (${path}): ${(err as Error).message}`);
  }
}

async function readBinary(path: string, label: string): Promise<ArrayBuffer> {
  const f = Bun.file(path);
  if (!(await f.exists())) {
    throw new Error(`missing ${label}: ${path}`);
  }
  return await f.arrayBuffer();
}

function expectByteLength(actual: number, expected: number, label: string): void {
  if (actual !== expected) {
    throw new Error(
      `${label} size mismatch: expected ${expected} bytes, got ${actual}`,
    );
  }
}

export async function loadIndex(dir: string): Promise<Index> {
  // 1) Header (catches schema mismatches before we touch the binaries).
  const header = await readJson<Header>(join(dir, "header.json"), "header.json");
  if (header.schemaVersion !== SCHEMA_VERSION) {
    throw new Error(
      `header.json: schemaVersion=${header.schemaVersion}, expected ${SCHEMA_VERSION}`,
    );
  }
  if (header.d !== D) {
    throw new Error(`header.json: d=${header.d}, expected ${D}`);
  }
  if (!Array.isArray(header.scale) || header.scale.length !== D) {
    throw new Error(`header.json: scale must be number[${D}]`);
  }
  const n = header.n;
  const k = header.k;
  // Allow runtime override of nprobe via env var so we can sweep without
  // rebuilding the image. Falls back to the build-time default.
  const envNprobe = process.env.NPROBE ? Number(process.env.NPROBE) : NaN;
  const nprobe = Number.isFinite(envNprobe) && envNprobe > 0
    ? Math.min(envNprobe, k)
    : header.nprobeDefault;
  if (!Number.isInteger(n) || n < 1) throw new Error(`header.json: bad n=${n}`);
  if (!Number.isInteger(k) || k < 1) throw new Error(`header.json: bad k=${k}`);
  if (!Number.isInteger(nprobe) || nprobe < 1 || nprobe > k) {
    throw new Error(`bad nprobe=${nprobe}`);
  }

  // 2) Read four binaries in parallel, then validate sizes.
  const [vBuf, lBuf, cBuf, oBuf] = await Promise.all([
    readBinary(join(dir, "vectors.i16"), "vectors.i16"),
    readBinary(join(dir, "labels.bits"), "labels.bits"),
    readBinary(join(dir, "centroids.f32"), "centroids.f32"),
    readBinary(join(dir, "offsets.u32"), "offsets.u32"),
  ]);
  expectByteLength(vBuf.byteLength, n * D * 2, "vectors.i16");
  expectByteLength(lBuf.byteLength, Math.ceil(n / 8), "labels.bits");
  expectByteLength(cBuf.byteLength, k * D * 4, "centroids.f32");
  expectByteLength(oBuf.byteLength, (k + 1) * 4, "offsets.u32");

  const vectors = new Int16Array(vBuf);
  const labels = new Uint8Array(lBuf);
  const centroids = new Float32Array(cBuf);
  const offsets = new Uint32Array(oBuf);

  // 3) Build scale + decodeFactor (allocates only 2*14 floats).
  const scale = new Float32Array(D);
  const decodeFactor = new Float32Array(D);
  for (let d = 0; d < D; d++) {
    const s = header.scale[d] as number;
    scale[d] = s;
    decodeFactor[d] = decodeFactorOf(s);
  }

  // 4) Reference JSON files.
  const mccRiskRaw = await readJson<Record<string, number>>(
    join(dir, "mcc_risk.json"),
    "mcc_risk.json",
  );
  const mccRisk = new Map<string, number>();
  for (const [key, value] of Object.entries(mccRiskRaw)) {
    if (typeof value === "number") mccRisk.set(key, value);
  }
  const norm = await readJson<NormConsts>(join(dir, "normalization.json"), "normalization.json");

  return {
    n,
    d: D,
    k,
    nprobe,
    scale,
    decodeFactor,
    vectors,
    labels,
    centroids,
    offsets,
    mccRisk,
    norm,
  };
}
