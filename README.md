# Rinha de Backend 2026 — Fraud Detection (Bun)

Pure-Bun TypeScript fraud-detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).
Hits sub-millisecond p99 search by precomputing an int8-quantized IVF index
at `docker build` time and serving lookups with zero per-request allocation.
Design spec: [`.omp/supipowers/specs/2026-05-05-rinha-fraud-detection-design.md`](./.omp/supipowers/specs/2026-05-05-rinha-fraud-detection-design.md).

## Layout

```
src/                 Runtime: Bun.serve, handlers, vectorize, IVF search, index loader
scripts/             Build-time: streaming gzip parser, kmeans, recall gate, CLI
bench/profile.ts     100k synthetic-query latency profiler
tests/               bun test suites (unit + integration + e2e)
nginx/nginx.conf     LB config with proxy_next_upstream and keepalive
Dockerfile           Builder runs preprocess.ts; runtime is bun:1-slim + binaries
docker-compose.yml   nginx LB + 2 Bun APIs, healthcheck-gated startup
resources/           (Not committed) reference data for the build (see below)
```

## Reference data

Place the contest's reference dataset and constants in `./resources/`:

- `resources/references.json.gz` — gzipped JSON array, each record
  `{ "vector": [number x 14], "label": "fraud" | "legit" }`
- `resources/normalization.json` — the `NormConsts` object from spec §6.1
- `resources/mcc_risk.json` — the MCC → risk-weight map

`resources/` is git-ignored. For a quick local smoke build use the synthetic
fixture generator:

```bash
bun scripts/make-synthetic-resources.ts 5000
```

## Local development

```bash
bun install
bun test                       # unit + integration + e2e
bun test tests/e2e.test.ts     # e2e only
bun run dev                    # boot src/server.ts on :8080
```

The bench profiler runs against any preprocessed `data/` directory:

```bash
DATA_DIR=./data BENCH_N=100000 bun bench/profile.ts
```

## Building and running the stack

```bash
# Build the image (runs preprocess.ts during the builder stage).
docker compose build

# Start lb + api1 + api2; lb is gated on both APIs reporting healthy.
docker compose up

# In another terminal:
curl -s http://localhost:9999/ready                # 200 {} when both APIs are loaded
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'content-type: application/json' \
  -d '{"transaction":{"amount":100,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},
       "customer":{"avg_amount":50,"tx_count_24h":3,"known_merchants":["M"]},
       "merchant":{"id":"M","mcc":"5411","avg_amount":60},
       "terminal":{"is_online":false,"card_present":true,"km_from_home":10},
       "last_transaction":null}'

docker compose down
```

## Tunables

`scripts/preprocess.ts` flags (override at build time via the `PREPROCESS_FLAGS`
build-arg):

| Flag | Default | Notes |
|---|---|---|
| `--k` | `2048` | Number of IVF cells (≈√N rounded to a power of two) |
| `--nprobe` | `4` | Cells probed per query at runtime |
| `--iters` | `25` | Mini-batch k-means iterations |
| `--batch` | `100000` | Mini-batch size |
| `--recall-floor` | `0.99` | Build fails if recall@5 falls below this |
| `--recall-sample` | `1000` | Number of sampled-reference recall queries |
| `--seed` | `42` | RNG seed for reproducible builds |

```bash
docker compose build --build-arg PREPROCESS_FLAGS="--k 4096 --nprobe 8"
```

## Acceptance: recall gate

Run preprocess against the real dataset before the official run:

```bash
bun scripts/preprocess.ts \
  --refs resources/references.json.gz \
  --norm resources/normalization.json \
  --mcc resources/mcc_risk.json \
  --out ./data-real
cat ./data-real/validation.txt   # recall@5 = …
```

If recall < 0.99, retry with `--k 4096`, then with `--k 4096 --nprobe 8`.
Bake the same flags into the Dockerfile via `PREPROCESS_FLAGS`.

## Acceptance: k6 load test

```bash
docker compose up --build           # in one terminal
# Wait until docker compose ps shows api1, api2 as "healthy"

k6 run rinha-test.js --env BASE_URL=http://localhost:9999
```

Acceptance for S5/S6 (spec §1):
- `http_req_failed.rate == 0` across 5xx, 4xx-on-valid, connection errors,
  and timeouts.
- `iteration_duration.p99 < 20ms` on a developer laptop (target: sub-ms p99 on
  contest hardware).

If anything fails, check `docker compose logs api1 api2` — the most likely
cause is a payload shape we did not anticipate. Add a fixture, fix
`vectorize` / `validatePayload`, re-run.

## Architecture quick reference

- One `Bun.serve` per API container, single event loop, no Worker threads
  (only one CPU is available compose-wide).
- Index is loaded once at boot into typed-array views over mmap-friendly
  files; per-request scratch buffers are reused.
- nginx LB on `:9999` is the only port exposed to the host; `:8080` is
  network-internal only.
- `docker-compose`'s `depends_on: condition: service_healthy` blocks `lb`
  startup until both APIs report healthy via their `/ready`-based
  healthcheck.
