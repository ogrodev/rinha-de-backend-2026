# syntax=docker/dockerfile:1.7
#
# Two-stage build:
#   1. builder — runs preprocess.ts against resources/references.json.gz to
#      materialize the IVF index binaries into /app/data. Peak builder memory
#      is ~250 MB (float buffer + int8 buffer + assignments). CI hosts MUST
#      have ≥512 MB available; the runtime 350 MB cap does not apply here.
#   2. runtime — slim Bun image with src + node_modules + the prebuilt /app/data.
#
# Build for the contest with: `docker build --platform linux/amd64 …`.
# This Dockerfile is otherwise platform-agnostic so local arm64 dev builds
# avoid amd64 emulation overhead.

# --- Builder ----------------------------------------------------------------

FROM oven/bun:1 AS builder
WORKDIR /app

# Lockfile-first install for reproducibility.
COPY package.json bun.lock bunfig.toml ./
RUN bun install --frozen-lockfile

# Preprocessing is now self-contained inside scripts/. Resources next, run.
COPY scripts ./scripts

# Reference data lives under ./resources in the build context.
#   - references.json.gz    (3M reference vectors, gzipped)
#   - normalization.json    (NormConsts)
#   - mcc_risk.json         (MCC -> risk map)
COPY resources ./resources

# Run preprocessing. Defaults match spec §3.4: K=2048, nprobe=4, iters=25,
# batch=100k, recall-floor=0.99. Override via build-args if Task 24 selects
# different tunables.
ARG PREPROCESS_FLAGS=""
RUN bun scripts/preprocess.ts \
  --refs resources/references.json.gz \
  --norm resources/normalization.json \
  --mcc resources/mcc_risk.json \
  --out /app/data \
  --nprobe 16 \
  ${PREPROCESS_FLAGS}

# --- Native SIMD search library ---------------------------------------------
# Compile after preprocess so it sits in its own cacheable layer. The C source
# rarely changes; this layer should hit the build cache on most rebuilds.
RUN apt-get update && apt-get install -y --no-install-recommends gcc make libc6-dev \
  && rm -rf /var/lib/apt/lists/*
COPY native ./native
RUN cd native && make

# --- Runtime ----------------------------------------------------------------

FROM oven/bun:1-slim AS runtime
WORKDIR /app

ENV DATA_DIR=/app/data \
    PORT=8080 \
    NODE_ENV=production

# `src/` is a runtime-only dependency now; copy directly from the build
# context (not from the builder stage) so changes don't bust the
# preprocess cache.
COPY src ./src
# Native FFI library (built in the builder stage).
COPY --from=builder /app/native/libsearch.so ./libsearch.so
COPY --from=builder /app/data ./data
COPY --from=builder /app/package.json /app/bunfig.toml ./
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 8080

# bun:1-slim has no curl/wget, so probe via Bun itself. /ready returns
# 503 while the index loads; --start-period gives that warming window.
HEALTHCHECK --interval=1s --timeout=2s --start-period=60s --retries=3 \
  CMD bun -e "fetch('http://127.0.0.1:8080/ready').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["bun", "src/server.ts"]
