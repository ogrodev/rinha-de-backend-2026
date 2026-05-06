// Single FFI entry point for the entire fraud-score hot path.
//
// Bun calls process_request(body_bytes, body_length) once per request and
// receives the fraud count [0..5] (or -1 on parse error) directly. This
// eliminates the JS-side JSON.parse, vectorize, and FFI cross from the path
// — the whole pipeline runs in C with one boundary crossing.
//
// JSON parser is hand-rolled for the contest payload shape and assumes:
//   - Whitespace is JSON-conforming but otherwise ignored.
//   - Field order within objects can vary.
//   - All numeric fields fit in float; integers fit in int.
//   - Strings are ASCII (MERC-IDs and MCC codes are ASCII; ISO-8601 is ASCII).

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define D 14

// External: defined in search.c
extern int32_t search_query(const float *q);

// Hardcoded normalization constants (locked at the contest's pinned values).
#define MAX_AMOUNT                10000.0f
#define MAX_INSTALLMENTS              12.0f
#define AMOUNT_VS_AVG_RATIO          10.0f
#define MAX_MINUTES                1440.0f
#define MAX_KM                     1000.0f
#define MAX_TX_COUNT_24H             20.0f
#define MAX_MERCHANT_AVG_AMOUNT   10000.0f

// Hardcoded MCC risk dict (10 entries, contest-locked).
typedef struct { const char *mcc; float risk; uint8_t len; } mcc_entry_t;
static const mcc_entry_t MCC_RISK[] = {
    {"5411", 0.15f, 4}, {"5812", 0.30f, 4}, {"5912", 0.20f, 4},
    {"5944", 0.45f, 4}, {"7801", 0.80f, 4}, {"7802", 0.75f, 4},
    {"7995", 0.85f, 4}, {"4511", 0.35f, 4}, {"5311", 0.25f, 4},
    {"5999", 0.50f, 4},
};
#define MCC_RISK_COUNT (sizeof(MCC_RISK) / sizeof(MCC_RISK[0]))

static inline float lookup_mcc(const char *s, int len) {
    for (size_t i = 0; i < MCC_RISK_COUNT; i++) {
        if (MCC_RISK[i].len == len && memcmp(MCC_RISK[i].mcc, s, len) == 0) {
            return MCC_RISK[i].risk;
        }
    }
    return 0.5f; // default per spec
}

static inline float clamp01(float x) {
    if (x != x) return NAN;          // NaN propagates
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

// --- Tiny JSON parser tailored to the contest payload ---

typedef struct {
    const char *p;
    const char *end;
} cur_t;

static inline void skip_ws(cur_t *c) {
    while (c->p < c->end) {
        char ch = *c->p;
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') c->p++;
        else break;
    }
}

static inline int expect(cur_t *c, char ch) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != ch) return -1;
    c->p++;
    return 0;
}

// Read a string between double quotes. Sets *out to the start of the string
// (inside the quotes) and *len to its length. Advances past the closing quote.
// Does NOT support escape sequences (contest payload doesn't use them).
static inline int read_string(cur_t *c, const char **out, int *len) {
    skip_ws(c);
    if (c->p >= c->end || *c->p != '"') return -1;
    c->p++;
    const char *s = c->p;
    while (c->p < c->end && *c->p != '"') c->p++;
    if (c->p >= c->end) return -1;
    *out = s;
    *len = (int)(c->p - s);
    c->p++;
    return 0;
}

// Read a JSON number (integer or float, optionally negative). Returns 0 and
// writes value to *out on success.
static inline int read_number(cur_t *c, double *out) {
    skip_ws(c);
    const char *start = c->p;
    if (c->p < c->end && (*c->p == '-' || *c->p == '+')) c->p++;
    while (c->p < c->end) {
        char ch = *c->p;
        if ((ch >= '0' && ch <= '9') || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-') {
            c->p++;
        } else break;
    }
    if (c->p == start) return -1;
    // Parse in place — strtod doesn't need null terminator if we restrict the slice via ptr math,
    // but it does. Copy into stack buffer for safety.
    char buf[64];
    int n = (int)(c->p - start);
    if (n >= (int)sizeof(buf)) return -1;
    memcpy(buf, start, n);
    buf[n] = 0;
    char *eptr;
    double v = strtod(buf, &eptr);
    if (eptr == buf) return -1;
    *out = v;
    return 0;
}

// Read a boolean (true/false). Returns 0 and writes *out on success.
static inline int read_bool(cur_t *c, int *out) {
    skip_ws(c);
    if (c->p + 4 <= c->end && memcmp(c->p, "true", 4) == 0) {
        c->p += 4; *out = 1; return 0;
    }
    if (c->p + 5 <= c->end && memcmp(c->p, "false", 5) == 0) {
        c->p += 5; *out = 0; return 0;
    }
    return -1;
}

// Read literal "null". Returns 0 if matched and consumed.
static inline int read_null(cur_t *c) {
    skip_ws(c);
    if (c->p + 4 <= c->end && memcmp(c->p, "null", 4) == 0) {
        c->p += 4; return 0;
    }
    return -1;
}

// Skip past a JSON value (any type, including nested arrays/objects).
// Used when we encounter unknown keys.
static int skip_value(cur_t *c) {
    skip_ws(c);
    if (c->p >= c->end) return -1;
    char ch = *c->p;
    if (ch == '"') {
        const char *s; int n;
        return read_string(c, &s, &n);
    }
    if (ch == '{') {
        c->p++;
        skip_ws(c);
        if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
        for (;;) {
            const char *k; int kn;
            if (read_string(c, &k, &kn) < 0) return -1;
            if (expect(c, ':') < 0) return -1;
            if (skip_value(c) < 0) return -1;
            skip_ws(c);
            if (c->p >= c->end) return -1;
            if (*c->p == ',') { c->p++; continue; }
            if (*c->p == '}') { c->p++; return 0; }
            return -1;
        }
    }
    if (ch == '[') {
        c->p++;
        skip_ws(c);
        if (c->p < c->end && *c->p == ']') { c->p++; return 0; }
        for (;;) {
            if (skip_value(c) < 0) return -1;
            skip_ws(c);
            if (c->p >= c->end) return -1;
            if (*c->p == ',') { c->p++; continue; }
            if (*c->p == ']') { c->p++; return 0; }
            return -1;
        }
    }
    if (ch == 't' || ch == 'f') {
        int b;
        return read_bool(c, &b);
    }
    if (ch == 'n') {
        return read_null(c);
    }
    // Number
    double v;
    return read_number(c, &v);
}

// --- ISO-8601 timestamp parsing (mirrors src/vectorize.ts) ---

static inline int is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
}

static const int8_t DAYS_IN_MONTH[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static const int8_t SAKAMOTO[12]     = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};

typedef struct { int y, m, d, h, mi, se; } ts_t;

static int parse_iso8601(const char *s, int len, ts_t *out) {
    if (len != 20) return -1;
    if (s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':' || s[16] != ':' || s[19] != 'Z') return -1;
    static const int dpos[14] = {0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18};
    for (int k = 0; k < 14; k++) {
        char c = s[dpos[k]];
        if (c < '0' || c > '9') return -1;
    }
    int y = (s[0]-'0')*1000 + (s[1]-'0')*100 + (s[2]-'0')*10 + (s[3]-'0');
    int m = (s[5]-'0')*10 + (s[6]-'0');
    int d = (s[8]-'0')*10 + (s[9]-'0');
    int h = (s[11]-'0')*10 + (s[12]-'0');
    int mi = (s[14]-'0')*10 + (s[15]-'0');
    int se = (s[17]-'0')*10 + (s[18]-'0');
    if (m < 1 || m > 12) return -1;
    if (h > 23 || mi > 59 || se > 59) return -1;
    int max_day = (m == 2 && is_leap(y)) ? 29 : DAYS_IN_MONTH[m - 1];
    if (d < 1 || d > max_day) return -1;
    out->y = y; out->m = m; out->d = d; out->h = h; out->mi = mi; out->se = se;
    return 0;
}

// Howard Hinnant days-from-civil epoch.
static inline int32_t days_from_civil(int y, int m, int d) {
    int yy = (m <= 2) ? y - 1 : y;
    int era = (yy >= 0) ? yy / 400 : (yy - 399) / 400;
    int yoe = yy - era * 400;
    int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static inline int32_t total_minutes(const ts_t *t) {
    return days_from_civil(t->y, t->m, t->d) * 1440 + t->h * 60 + t->mi;
}

static inline int dow_mon0(int y, int m, int d) {
    int yy = y;
    if (m < 3) yy -= 1;
    int dow_sun = (yy + yy/4 - yy/100 + yy/400 + SAKAMOTO[m - 1] + d) % 7;
    return (dow_sun + 6) % 7;
}

// --- Field name dispatch ---

// Match a field name against known keys. Returns a small enum-like int.
// FIELD_UNKNOWN = -1.
typedef enum {
    F_UNKNOWN = -1,
    F_AMOUNT, F_INSTALLMENTS, F_REQUESTED_AT,
    F_AVG_AMOUNT, F_TX_COUNT_24H, F_KNOWN_MERCHANTS,
    F_ID, F_MCC,
    F_IS_ONLINE, F_CARD_PRESENT, F_KM_FROM_HOME,
    F_TIMESTAMP, F_KM_FROM_CURRENT,
    F_TRANSACTION, F_CUSTOMER, F_MERCHANT, F_TERMINAL, F_LAST_TRANSACTION,
} field_t;

static field_t classify(const char *k, int n) {
    switch (n) {
        case 2: if (memcmp(k, "id", 2) == 0) return F_ID; break;
        case 3: if (memcmp(k, "mcc", 3) == 0) return F_MCC; break;
        case 6: if (memcmp(k, "amount", 6) == 0) return F_AMOUNT; break;
        case 8: if (memcmp(k, "customer", 8) == 0) return F_CUSTOMER;
                if (memcmp(k, "merchant", 8) == 0) return F_MERCHANT;
                if (memcmp(k, "terminal", 8) == 0) return F_TERMINAL; break;
        case 9: if (memcmp(k, "is_online", 9) == 0) return F_IS_ONLINE;
                if (memcmp(k, "timestamp", 9) == 0) return F_TIMESTAMP; break;
        case 10: if (memcmp(k, "avg_amount", 10) == 0) return F_AVG_AMOUNT; break;
        case 11: if (memcmp(k, "transaction", 11) == 0) return F_TRANSACTION;
                 if (memcmp(k, "installments", 12) == 0) return F_INSTALLMENTS; break;
        case 12: if (memcmp(k, "tx_count_24h", 12) == 0) return F_TX_COUNT_24H;
                 if (memcmp(k, "requested_at", 12) == 0) return F_REQUESTED_AT;
                 if (memcmp(k, "installments", 12) == 0) return F_INSTALLMENTS;
                 if (memcmp(k, "km_from_home", 12) == 0) return F_KM_FROM_HOME;
                 if (memcmp(k, "card_present", 12) == 0) return F_CARD_PRESENT; break;
        case 15: if (memcmp(k, "km_from_current", 15) == 0) return F_KM_FROM_CURRENT;
                 if (memcmp(k, "known_merchants", 15) == 0) return F_KNOWN_MERCHANTS; break;
        case 16: if (memcmp(k, "last_transaction", 16) == 0) return F_LAST_TRANSACTION; break;
    }
    return F_UNKNOWN;
}

// --- Payload struct + sub-object parsers ---

typedef struct {
    float    tx_amount;
    int      tx_installments;
    ts_t     tx_requested_at;
    int      tx_requested_at_ok;

    float    cust_avg_amount;
    int      cust_tx_count_24h;
    // For known_merchants: store packed strings (offset+len) into a side buffer.
    int      known_merchants_count;
    const char *known_merchants[16];
    int      known_merchants_len[16];

    const char *mer_id;
    int      mer_id_len;
    const char *mer_mcc;
    int      mer_mcc_len;
    float    mer_avg_amount;

    int      term_is_online;
    int      term_card_present;
    float    term_km_from_home;

    int      last_present;
    ts_t     last_timestamp;
    int      last_timestamp_ok;
    float    last_km_from_current;
} payload_t;

static int parse_transaction(cur_t *c, payload_t *p) {
    if (expect(c, '{') < 0) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    for (;;) {
        const char *k; int kn;
        if (read_string(c, &k, &kn) < 0) return -1;
        if (expect(c, ':') < 0) return -1;
        field_t f = classify(k, kn);
        if (f == F_AMOUNT) {
            double v; if (read_number(c, &v) < 0) return -1; p->tx_amount = (float)v;
        } else if (f == F_INSTALLMENTS) {
            double v; if (read_number(c, &v) < 0) return -1; p->tx_installments = (int)v;
        } else if (f == F_REQUESTED_AT) {
            const char *s; int n;
            if (read_string(c, &s, &n) < 0) return -1;
            p->tx_requested_at_ok = (parse_iso8601(s, n, &p->tx_requested_at) == 0);
            if (!p->tx_requested_at_ok) return -1;
        } else {
            if (skip_value(c) < 0) return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return 0; }
        return -1;
    }
}

static int parse_customer(cur_t *c, payload_t *p) {
    if (expect(c, '{') < 0) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    for (;;) {
        const char *k; int kn;
        if (read_string(c, &k, &kn) < 0) return -1;
        if (expect(c, ':') < 0) return -1;
        field_t f = classify(k, kn);
        if (f == F_AVG_AMOUNT) {
            double v; if (read_number(c, &v) < 0) return -1; p->cust_avg_amount = (float)v;
        } else if (f == F_TX_COUNT_24H) {
            double v; if (read_number(c, &v) < 0) return -1; p->cust_tx_count_24h = (int)v;
        } else if (f == F_KNOWN_MERCHANTS) {
            if (expect(c, '[') < 0) return -1;
            skip_ws(c);
            if (c->p < c->end && *c->p == ']') { c->p++; }
            else {
                for (;;) {
                    const char *s; int n;
                    if (read_string(c, &s, &n) < 0) return -1;
                    if (p->known_merchants_count < 16) {
                        p->known_merchants[p->known_merchants_count] = s;
                        p->known_merchants_len[p->known_merchants_count] = n;
                        p->known_merchants_count++;
                    }
                    skip_ws(c);
                    if (c->p >= c->end) return -1;
                    if (*c->p == ',') { c->p++; continue; }
                    if (*c->p == ']') { c->p++; break; }
                    return -1;
                }
            }
        } else {
            if (skip_value(c) < 0) return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return 0; }
        return -1;
    }
}

static int parse_merchant(cur_t *c, payload_t *p) {
    if (expect(c, '{') < 0) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    for (;;) {
        const char *k; int kn;
        if (read_string(c, &k, &kn) < 0) return -1;
        if (expect(c, ':') < 0) return -1;
        field_t f = classify(k, kn);
        if (f == F_ID) {
            if (read_string(c, &p->mer_id, &p->mer_id_len) < 0) return -1;
        } else if (f == F_MCC) {
            if (read_string(c, &p->mer_mcc, &p->mer_mcc_len) < 0) return -1;
        } else if (f == F_AVG_AMOUNT) {
            double v; if (read_number(c, &v) < 0) return -1; p->mer_avg_amount = (float)v;
        } else {
            if (skip_value(c) < 0) return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return 0; }
        return -1;
    }
}

static int parse_terminal(cur_t *c, payload_t *p) {
    if (expect(c, '{') < 0) return -1;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    for (;;) {
        const char *k; int kn;
        if (read_string(c, &k, &kn) < 0) return -1;
        if (expect(c, ':') < 0) return -1;
        field_t f = classify(k, kn);
        if (f == F_IS_ONLINE) {
            if (read_bool(c, &p->term_is_online) < 0) return -1;
        } else if (f == F_CARD_PRESENT) {
            if (read_bool(c, &p->term_card_present) < 0) return -1;
        } else if (f == F_KM_FROM_HOME) {
            double v; if (read_number(c, &v) < 0) return -1; p->term_km_from_home = (float)v;
        } else {
            if (skip_value(c) < 0) return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return 0; }
        return -1;
    }
}

static int parse_last_transaction(cur_t *c, payload_t *p) {
    skip_ws(c);
    if (c->p >= c->end) return -1;
    if (*c->p == 'n') {
        if (read_null(c) < 0) return -1;
        p->last_present = 0;
        return 0;
    }
    if (expect(c, '{') < 0) return -1;
    p->last_present = 1;
    skip_ws(c);
    if (c->p < c->end && *c->p == '}') { c->p++; return 0; }
    for (;;) {
        const char *k; int kn;
        if (read_string(c, &k, &kn) < 0) return -1;
        if (expect(c, ':') < 0) return -1;
        field_t f = classify(k, kn);
        if (f == F_TIMESTAMP) {
            const char *s; int n;
            if (read_string(c, &s, &n) < 0) return -1;
            p->last_timestamp_ok = (parse_iso8601(s, n, &p->last_timestamp) == 0);
            if (!p->last_timestamp_ok) return -1;
        } else if (f == F_KM_FROM_CURRENT) {
            double v; if (read_number(c, &v) < 0) return -1; p->last_km_from_current = (float)v;
        } else {
            if (skip_value(c) < 0) return -1;
        }
        skip_ws(c);
        if (c->p >= c->end) return -1;
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return 0; }
        return -1;
    }
}

// Top-level parser.
static int parse_payload(const char *body, int body_len, payload_t *p) {
    memset(p, 0, sizeof(*p));
    cur_t c = { body, body + body_len };
    if (expect(&c, '{') < 0) return -1;
    skip_ws(&c);
    if (c.p < c.end && *c.p == '}') return 0;
    for (;;) {
        const char *k; int kn;
        if (read_string(&c, &k, &kn) < 0) return -1;
        if (expect(&c, ':') < 0) return -1;
        field_t f = classify(k, kn);
        switch (f) {
            case F_TRANSACTION:      if (parse_transaction(&c, p) < 0) return -1; break;
            case F_CUSTOMER:         if (parse_customer(&c, p) < 0) return -1; break;
            case F_MERCHANT:         if (parse_merchant(&c, p) < 0) return -1; break;
            case F_TERMINAL:         if (parse_terminal(&c, p) < 0) return -1; break;
            case F_LAST_TRANSACTION: if (parse_last_transaction(&c, p) < 0) return -1; break;
            default:                 if (skip_value(&c) < 0) return -1;
        }
        skip_ws(&c);
        if (c.p >= c.end) return -1;
        if (*c.p == ',') { c.p++; continue; }
        if (*c.p == '}') { c.p++; return 0; }
        return -1;
    }
}

// --- Vectorize + search (all in C) ---

int32_t process_request(const char *body, int32_t body_len, float *out) {
    payload_t p;
    if (parse_payload(body, body_len, &p) != 0) return -1;
    if (!p.tx_requested_at_ok) return -1;

    // dim 0..4
    out[0] = clamp01(p.tx_amount / MAX_AMOUNT);
    out[1] = clamp01((float)p.tx_installments / MAX_INSTALLMENTS);
    out[2] = clamp01(p.tx_amount / p.cust_avg_amount / AMOUNT_VS_AVG_RATIO);
    out[3] = (float)p.tx_requested_at.h / 23.0f;
    out[4] = (float)dow_mon0(p.tx_requested_at.y, p.tx_requested_at.m, p.tx_requested_at.d) / 6.0f;

    // dim 5, 6: last_transaction-derived (or -1).
    if (!p.last_present) {
        out[5] = -1.0f;
        out[6] = -1.0f;
    } else {
        if (!p.last_timestamp_ok) return -1;
        int32_t delta = total_minutes(&p.tx_requested_at) - total_minutes(&p.last_timestamp);
        out[5] = clamp01((float)delta / MAX_MINUTES);
        out[6] = clamp01(p.last_km_from_current / MAX_KM);
    }

    // dim 7..10
    out[7]  = clamp01(p.term_km_from_home / MAX_KM);
    out[8]  = clamp01((float)p.cust_tx_count_24h / MAX_TX_COUNT_24H);
    out[9]  = p.term_is_online ? 1.0f : 0.0f;
    out[10] = p.term_card_present ? 1.0f : 0.0f;

    // dim 11: unknown_merchant
    int known = 0;
    for (int i = 0; i < p.known_merchants_count; i++) {
        if (p.known_merchants_len[i] == p.mer_id_len &&
            memcmp(p.known_merchants[i], p.mer_id, p.mer_id_len) == 0) {
            known = 1; break;
        }
    }
    out[11] = known ? 0.0f : 1.0f;

    // dim 12: mcc_risk
    out[12] = lookup_mcc(p.mer_mcc, p.mer_mcc_len);

    // dim 13: merchant_avg_amount
    out[13] = clamp01(p.mer_avg_amount / MAX_MERCHANT_AVG_AMOUNT);

    // NaN scan.
    for (int d = 0; d < D; d++) {
        if (out[d] != out[d]) return -1;
    }

    return search_query(out);
}
