import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { handleReady, handleFraudScore, type AppState } from "../src/handlers.ts";
import { loadIndex } from "../src/index/load.ts";
import { makeScratch } from "../src/index/search.ts";
import {
  buildScale,
  quantizeAll,
  clusterSort,
} from "../scripts/preprocess.ts";
import { miniBatchKMeans, lcg } from "../scripts/kmeans.ts";

const D = 14;

async function buildAllFraudIndex(): Promise<AppState> {
  // Tiny fraud-only synthetic index: every label = 1, so any top-5 yields 1.0.
  const n = 1024;
  const K = 16;
  const flat = new Float32Array(n * D);
  const labels = new Uint8Array(Math.ceil(n / 8));
  const rng = lcg(2024);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) flat[i * D + d] = (rng() - 0.5) * 2;
    labels[i >> 3]! |= 1 << (i & 7); // every record is fraud
  }
  const scale = buildScale(flat, n, D);
  const i8 = quantizeAll(flat, scale, n, D);
  const { centroids, assignments } = miniBatchKMeans(flat, n, D, K, 5, 200, 1);
  const { sortedVectors, sortedLabels, offsets } = clusterSort(
    i8,
    labels,
    assignments,
    n,
    D,
    K,
  );

  const dir = mkdtempSync(join(tmpdir(), "rinha-handlers-"));
  await Bun.write(
    join(dir, "header.json"),
    JSON.stringify({ n, d: 14, k: K, nprobeDefault: 4, scale: Array.from(scale), schemaVersion: 2 }),
  );
  await Bun.write(join(dir, "vectors.i16"), sortedVectors);
  await Bun.write(join(dir, "labels.bits"), sortedLabels);
  await Bun.write(
    join(dir, "centroids.f32"),
    new Uint8Array(centroids.buffer, centroids.byteOffset, centroids.byteLength),
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

  const idx = await loadIndex(dir);
  return {
    ready: true,
    idx,
    scratch: makeScratch(idx.k, idx.nprobe),
    queryBuf: new Float32Array(D),
  };
}

const VALID_PAYLOAD = {
  transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
  customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["MERC-001"] },
  merchant: { id: "MERC-002", mcc: "5411", avg_amount: 60 },
  terminal: { is_online: false, card_present: true, km_from_home: 10 },
  last_transaction: null,
};

function jsonReq(body: unknown): Request {
  return new Request("http://x/fraud-score", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("handleReady", () => {
  test("returns 503 not_ready when state.ready === false", async () => {
    const res = handleReady({ ready: false });
    expect(res.status).toBe(503);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.text()).toBe('{"error":"not_ready"}');
  });

  test("returns 200 {} when state.ready === true", async () => {
    const res = handleReady({ ready: true });
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.text()).toBe("{}");
  });
});

describe("handleFraudScore", () => {
  test("503 not_ready when state is missing the index — body is NOT read", async () => {
    // A Request with a body that throws on .json() to assert no parse happens.
    const req = new Request("http://x/fraud-score", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "not-json-at-all",
    });
    const res = await handleFraudScore(req, { ready: false });
    expect(res.status).toBe(503);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.text()).toBe('{"error":"not_ready"}');
  });

  test("400 invalid_json on malformed JSON body", async () => {
    const state = await buildAllFraudIndex();
    const req = new Request("http://x/fraud-score", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{ not valid",
    });
    const res = await handleFraudScore(req, state);
    expect(res.status).toBe(400);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.text()).toBe('{"error":"invalid_json"}');
  });

  test("missing merchant.mcc falls through to default 0.5 (no validation overhead in hot path)", async () => {
    const state = await buildAllFraudIndex();
    const broken = { ...VALID_PAYLOAD, merchant: { id: "M", mcc: "9999", avg_amount: 60 } };
    const res = await handleFraudScore(jsonReq(broken), state);
    // Unknown MCC just defaults to 0.5; vectorize succeeds.
    expect(res.status).toBe(200);
  });

  test("200 with hot-path body shape on a valid all-fraud payload", async () => {
    const state = await buildAllFraudIndex();
    const res = await handleFraudScore(jsonReq(VALID_PAYLOAD), state);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.text()).toBe('{"approved":false,"fraud_score":1}');
  });
});
