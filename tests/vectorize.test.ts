import { describe, expect, test } from "bun:test";
import { vectorize } from "../src/vectorize.ts";
import type { NormConsts, TxPayload } from "../src/index/types.ts";

const NORM: NormConsts = {
  max_amount: 10000,
  max_installments: 12,
  amount_vs_avg_ratio: 10,
  max_minutes: 1440,
  max_km: 1000,
  max_tx_count_24h: 20,
  max_merchant_avg_amount: 10000,
};

const MCC_RISK = new Map<string, number>([
  ["5411", 0.15],
  ["7802", 0.75],
]);

function vec(payload: TxPayload): Float32Array {
  const out = new Float32Array(14);
  const ok = vectorize(payload, NORM, MCC_RISK, out);
  if (!ok) throw new Error("vectorize returned false");
  return out;
}

function assertVec(actual: Float32Array, expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs((actual[i] as number) - (expected[i] as number))).toBeLessThan(5e-4);
  }
}

describe("vectorize: golden flow examples", () => {
  test("Fixture A — legitimate transaction (last_transaction=null)", () => {
    const payload: TxPayload = {
      transaction: { amount: 41.12, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
      customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ["MERC-003", "MERC-016"] },
      merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
      terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
      last_transaction: null,
    };
    const out = vec(payload);
    assertVec(out, [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]);
  });

  test("Fixture B — fraudulent transaction (last_transaction=null)", () => {
    const payload: TxPayload = {
      transaction: { amount: 9505.97, installments: 10, requested_at: "2026-03-14T05:15:12Z" },
      customer: { avg_amount: 81.28, tx_count_24h: 20, known_merchants: ["MERC-008", "MERC-007", "MERC-005"] },
      merchant: { id: "MERC-068", mcc: "7802", avg_amount: 54.86 },
      terminal: { is_online: false, card_present: true, km_from_home: 952.27 },
      last_transaction: null,
    };
    const out = vec(payload);
    assertVec(out, [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]);
  });
});

describe("vectorize: edge cases", () => {
  function basePayload(overrides: Partial<TxPayload> = {}): TxPayload {
    return {
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
      customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["MERC-001"] },
      merchant: { id: "MERC-001", mcc: "5411", avg_amount: 60 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
      ...overrides,
    };
  }

  test("last_transaction=null sets dims 5,6 to -1", () => {
    const out = vec(basePayload({ last_transaction: null }));
    expect(out[5]).toBe(-1);
    expect(out[6]).toBe(-1);
  });

  test("last_transaction same-day → dim 5 = minutes/1440", () => {
    const out = vec(basePayload({
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T18:45:00Z" },
      last_transaction: { timestamp: "2026-03-11T18:30:00Z", km_from_current: 0 },
    }));
    // 15 minutes / 1440 = 0.01041666...
    expect(Math.abs((out[5] as number) - 15 / 1440)).toBeLessThan(1e-5);
  });

  test("last_transaction cross-day → dim 5 picks up day boundary", () => {
    const out = vec(basePayload({
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T00:10:00Z" },
      last_transaction: { timestamp: "2026-03-10T23:50:00Z", km_from_current: 0 },
    }));
    // 20 minutes / 1440
    expect(Math.abs((out[5] as number) - 20 / 1440)).toBeLessThan(1e-5);
  });

  test("last_transaction far in past → dim 5 saturates at 1", () => {
    const out = vec(basePayload({
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T00:00:00Z" },
      last_transaction: { timestamp: "2026-03-01T00:00:00Z", km_from_current: 0 },
    }));
    expect(out[5]).toBe(1);
  });

  test("missing MCC → dim 12 = 0.5", () => {
    const out = vec(basePayload({
      merchant: { id: "MERC-001", mcc: "9999", avg_amount: 60 },
    }));
    expect(out[12]).toBeCloseTo(0.5, 6);
  });

  test("customer.avg_amount=0 → dim 2 = 1.0 (clamp01(Infinity))", () => {
    const out = vec(basePayload({
      customer: { avg_amount: 0, tx_count_24h: 3, known_merchants: ["MERC-001"] },
    }));
    expect(out[2]).toBe(1);
  });

  test("Monday 00:00 UTC → dims 3,4 = 0", () => {
    const out = vec(basePayload({
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-09T00:00:00Z" },
    }));
    expect(out[3]).toBe(0);
    expect(out[4]).toBe(0);
  });

  test("Sunday 23:00 UTC → dims 3,4 = 1", () => {
    const out = vec(basePayload({
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-15T23:00:00Z" },
    }));
    expect(out[3]).toBe(1);
    expect(out[4]).toBe(1);
  });

  test("merchant.id in known_merchants → dim 11 = 0", () => {
    const out = vec(basePayload({
      customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["MERC-A", "MERC-B"] },
      merchant: { id: "MERC-A", mcc: "5411", avg_amount: 60 },
    }));
    expect(out[11]).toBe(0);
  });

  test("merchant.id NOT in known_merchants → dim 11 = 1", () => {
    const out = vec(basePayload({
      customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["MERC-A", "MERC-B"] },
      merchant: { id: "MERC-Z", mcc: "5411", avg_amount: 60 },
    }));
    expect(out[11]).toBe(1);
  });

  test("terminal.is_online=true → dim 9 = 1", () => {
    const out = vec(basePayload({
      terminal: { is_online: true, card_present: true, km_from_home: 10 },
    }));
    expect(out[9]).toBe(1);
  });

  test("terminal.card_present=false → dim 10 = 0", () => {
    const out = vec(basePayload({
      terminal: { is_online: false, card_present: false, km_from_home: 10 },
    }));
    expect(out[10]).toBe(0);
  });
});

describe("vectorize: malformed timestamps return false", () => {
  function makePayload(reqAt: string): TxPayload {
    return {
      transaction: { amount: 100, installments: 2, requested_at: reqAt },
      customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["M"] },
      merchant: { id: "M", mcc: "5411", avg_amount: 60 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: null,
    };
  }

  const bad = [
    "not-a-date",
    "2026-03-11T18:45:53",       // missing Z, length 19
    "2026-03-11 18:45:53Z",      // space instead of T
    "2026-13-11T18:45:53Z",      // month 13
    "2026-03-32T18:45:53Z",      // day 32
    "2026-04-31T18:45:53Z",      // April only has 30 days
    "2026-02-29T18:45:53Z",      // 2026 is not a leap year
    "2024-02-30T18:45:53Z",      // even leap year doesn't allow Feb 30
    "2026-03-11T24:00:00Z",      // hour 24
    "2026-03-11T18:60:00Z",      // minute 60
    "2026-03-11T18:45:60Z",      // second 60
    "2026-03-11T1A:45:53Z",      // non-digit in hour
  ];

  for (const s of bad) {
    test(`rejects ${JSON.stringify(s)}`, () => {
      const out = new Float32Array(14);
      const ok = vectorize(makePayload(s), NORM, MCC_RISK, out);
      expect(ok).toBe(false);
    });
  }

  test("accepts 2024-02-29 (leap year)", () => {
    const out = new Float32Array(14);
    const ok = vectorize(makePayload("2024-02-29T00:00:00Z"), NORM, MCC_RISK, out);
    expect(ok).toBe(true);
  });
});

describe("vectorize: allocation stability", () => {
  test("10k calls with reused buffer leak <1MB", () => {
    const out = new Float32Array(14);
    const payload: TxPayload = {
      transaction: { amount: 100, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
      customer: { avg_amount: 50, tx_count_24h: 3, known_merchants: ["M-1", "M-2", "M-3"] },
      merchant: { id: "M-2", mcc: "5411", avg_amount: 60 },
      terminal: { is_online: false, card_present: true, km_from_home: 10 },
      last_transaction: { timestamp: "2026-03-11T18:30:00Z", km_from_current: 5 },
    };
    if (typeof Bun !== "undefined" && (Bun as any).gc) (Bun as any).gc(true);
    const before = process.memoryUsage().heapUsed;
    for (let n = 0; n < 10_000; n++) {
      vectorize(payload, NORM, MCC_RISK, out);
    }
    const after = process.memoryUsage().heapUsed;
    expect(after - before).toBeLessThan(1024 * 1024);
  });
});
