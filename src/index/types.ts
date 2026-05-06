// Shared runtime types. No runtime code lives here.
//
// `Index` is finalized in Task 14 (Index loader). For Task 6 we only need
// `NormConsts` and `TxPayload`; the rest is added together with the loader.

export type NormConsts = {
  readonly max_amount: number;
  readonly max_installments: number;
  readonly amount_vs_avg_ratio: number;
  readonly max_minutes: number;
  readonly max_km: number;
  readonly max_tx_count_24h: number;
  readonly max_merchant_avg_amount: number;
};

export type TxPayload = {
  readonly transaction: {
    readonly amount: number;
    readonly installments: number;
    readonly requested_at: string;
  };
  readonly customer: {
    readonly avg_amount: number;
    readonly tx_count_24h: number;
    readonly known_merchants: ReadonlyArray<string>;
  };
  readonly merchant: {
    readonly id: string;
    readonly mcc: string;
    readonly avg_amount: number;
  };
  readonly terminal: {
    readonly is_online: boolean;
    readonly card_present: boolean;
    readonly km_from_home: number;
  };
  readonly last_transaction:
    | null
    | {
        readonly timestamp: string;
        readonly km_from_current: number;
      };
};

// `Index` is the loaded runtime data. All typed-array fields are zero-copy
// views over the same underlying ArrayBuffers as the on-disk binaries.
export type Index = {
  readonly n: number;
  readonly d: 14;
  readonly k: number;
  readonly nprobe: number;
  readonly scale: Float32Array;
  readonly decodeFactor: Float32Array;
  readonly vectors: Int16Array;
  readonly labels: Uint8Array;
  readonly centroids: Float32Array;
  readonly offsets: Uint32Array;
  readonly mccRisk: Map<string, number>;
  readonly norm: NormConsts;
};
