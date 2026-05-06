// Vectorize a fraud-score payload into a 14-dim float buffer (spec §6.1).
//
// Contract:
//   - `out` MUST be a Float32Array of length 14, owned by the caller.
//   - Returns `true` if the buffer was filled successfully; `false` if the
//     `requested_at` (or `last_transaction.timestamp`) is malformed, or if
//     any computed dimension came out NaN.
//   - Zero allocation per call: timestamps are parsed via direct `charCodeAt`
//     against the fixed-width ISO-8601 layout `YYYY-MM-DDTHH:MM:SSZ`, day-of-
//     week is computed via Sakamoto's algorithm with integer arithmetic, and
//     time deltas use Howard Hinnant's `daysFromCivil` formula. No `Date`
//     objects are created.

import { clamp01 } from "./util/clamp.ts";
import type { NormConsts, TxPayload } from "./index/types.ts";

// --- Module-level scratch (allocated once at import) ---

type DateScratch = { y: number; m: number; d: number; h: number; mi: number; se: number };
const SCRATCH_CUR: DateScratch = { y: 0, m: 0, d: 0, h: 0, mi: 0, se: 0 };
const SCRATCH_PREV: DateScratch = { y: 0, m: 0, d: 0, h: 0, mi: 0, se: 0 };

// Days in each month for non-leap years (1-indexed via [m-1]).
const DAYS_IN_MONTH = Int8Array.from([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]);

// Sakamoto month table. Returns 0=Sunday..6=Saturday for the given (y, m, d).
const SAKAMOTO = Int8Array.from([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]);

// --- Timestamp parsing ---

function isLeap(y: number): boolean {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

// Parse "YYYY-MM-DDTHH:MM:SSZ" into `out`. Returns false on any divergence.
// Validates calendar-correct day-of-month (rejects Feb 30, Apr 31, etc.).
function parseIso(s: string, out: DateScratch): boolean {
  if (s.length !== 20) return false;
  // Static separators.
  if (s.charCodeAt(4) !== 45) return false;   // '-'
  if (s.charCodeAt(7) !== 45) return false;   // '-'
  if (s.charCodeAt(10) !== 84) return false;  // 'T'
  if (s.charCodeAt(13) !== 58) return false;  // ':'
  if (s.charCodeAt(16) !== 58) return false;  // ':'
  if (s.charCodeAt(19) !== 90) return false;  // 'Z'

  // Digit positions: 0..3, 5..6, 8..9, 11..12, 14..15, 17..18.
  const digits: number[] = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
  for (let k = 0; k < digits.length; k++) {
    const c = s.charCodeAt(digits[k] as number);
    if (c < 48 || c > 57) return false;
  }

  const y =
    (s.charCodeAt(0) - 48) * 1000 +
    (s.charCodeAt(1) - 48) * 100 +
    (s.charCodeAt(2) - 48) * 10 +
    (s.charCodeAt(3) - 48);
  const m = (s.charCodeAt(5) - 48) * 10 + (s.charCodeAt(6) - 48);
  const d = (s.charCodeAt(8) - 48) * 10 + (s.charCodeAt(9) - 48);
  const h = (s.charCodeAt(11) - 48) * 10 + (s.charCodeAt(12) - 48);
  const mi = (s.charCodeAt(14) - 48) * 10 + (s.charCodeAt(15) - 48);
  const se = (s.charCodeAt(17) - 48) * 10 + (s.charCodeAt(18) - 48);

  if (m < 1 || m > 12) return false;
  if (h > 23) return false;
  if (mi > 59) return false;
  if (se > 59) return false;

  const maxDay = m === 2 && isLeap(y) ? 29 : (DAYS_IN_MONTH[m - 1] as number);
  if (d < 1 || d > maxDay) return false;

  out.y = y;
  out.m = m;
  out.d = d;
  out.h = h;
  out.mi = mi;
  out.se = se;
  return true;
}

// Howard Hinnant's days-from-civil. Returns the count of days since the civil
// epoch 1970-01-01 (negative for earlier dates). Pure integer arithmetic, no
// allocation. Algorithm reference: https://howardhinnant.github.io/date_algorithms.html
function daysFromCivil(y: number, m: number, d: number): number {
  const yy = m <= 2 ? y - 1 : y;
  const era = Math.floor(yy / 400);
  const yoe = yy - era * 400; // [0, 399]
  // doy in [0, 365]
  const doy = (((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) | 0) + d - 1;
  const doe = yoe * 365 + ((yoe / 4) | 0) - ((yoe / 100) | 0) + doy;
  return era * 146097 + doe - 719468;
}

function totalMinutes(s: DateScratch): number {
  return daysFromCivil(s.y, s.m, s.d) * 1440 + s.h * 60 + s.mi;
}

// Sakamoto, returning Mon=0..Sun=6.
function dayOfWeekMon0(y: number, m: number, d: number): number {
  let yy = y;
  if (m < 3) yy -= 1;
  const dowSun =
    (yy +
      ((yy / 4) | 0) -
      ((yy / 100) | 0) +
      ((yy / 400) | 0) +
      (SAKAMOTO[m - 1] as number) +
      d) %
    7;
  return (dowSun + 6) % 7;
}

// --- Vectorize ---

export function vectorize(
  payload: TxPayload,
  norm: NormConsts,
  mccRisk: Map<string, number>,
  out: Float32Array,
): boolean {
  const tx = payload.transaction;
  if (!parseIso(tx.requested_at, SCRATCH_CUR)) return false;

  // 0: amount
  out[0] = clamp01(tx.amount / norm.max_amount);
  // 1: installments
  out[1] = clamp01(tx.installments / norm.max_installments);
  // 2: amount_vs_avg — divide-by-zero yields Infinity which clamps to 1.0.
  out[2] = clamp01(tx.amount / payload.customer.avg_amount / norm.amount_vs_avg_ratio);
  // 3: hour_of_day (UTC integer hour / 23)
  out[3] = SCRATCH_CUR.h / 23;
  // 4: day_of_week (Mon=0..Sun=6) / 6
  out[4] = dayOfWeekMon0(SCRATCH_CUR.y, SCRATCH_CUR.m, SCRATCH_CUR.d) / 6;

  // 5, 6: last_transaction-derived (or -1 if null).
  const last = payload.last_transaction;
  if (last === null) {
    out[5] = -1;
    out[6] = -1;
  } else {
    if (!parseIso(last.timestamp, SCRATCH_PREV)) return false;
    const delta = totalMinutes(SCRATCH_CUR) - totalMinutes(SCRATCH_PREV);
    out[5] = clamp01(delta / norm.max_minutes);
    out[6] = clamp01(last.km_from_current / norm.max_km);
  }

  // 7: km_from_home
  out[7] = clamp01(payload.terminal.km_from_home / norm.max_km);
  // 8: tx_count_24h
  out[8] = clamp01(payload.customer.tx_count_24h / norm.max_tx_count_24h);
  // 9: is_online
  out[9] = payload.terminal.is_online ? 1 : 0;
  // 10: card_present
  out[10] = payload.terminal.card_present ? 1 : 0;

  // 11: unknown_merchant — tight loop instead of Array.includes to avoid
  // call-site overhead and stay deterministic on hot path.
  const known = payload.customer.known_merchants;
  const merchantId = payload.merchant.id;
  let isKnown = 0;
  for (let i = 0; i < known.length; i++) {
    if (known[i] === merchantId) {
      isKnown = 1;
      break;
    }
  }
  out[11] = isKnown ? 0 : 1;

  // 12: mcc_risk — default 0.5 per spec.
  const mcc = mccRisk.get(payload.merchant.mcc);
  out[12] = mcc === undefined ? 0.5 : mcc;

  // 13: merchant_avg_amount
  out[13] = clamp01(payload.merchant.avg_amount / norm.max_merchant_avg_amount);

  // Final NaN scan — clamp01 propagates NaN from any malformed numeric.
  for (let d = 0; d < 14; d++) {
    if (out[d] !== out[d]) return false;
  }
  return true;
}
