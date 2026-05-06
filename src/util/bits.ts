// Read bit `i` from an LSB-first packed Uint8Array.
//
// Layout: bit `i` of the conceptual stream lives at byte `i >> 3`, position
// `i & 7` (least-significant first). Caller MUST ensure `0 <= i < buf.length * 8`;
// no bounds checking is performed because this sits on the hot path.
export function getBit(buf: Uint8Array, i: number): 0 | 1 {
  return ((buf[i >> 3]! >> (i & 7)) & 1) as 0 | 1;
}
