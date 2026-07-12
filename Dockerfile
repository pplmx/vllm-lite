# Multi-stage Dockerfile for vLLM-lite.
#
# DEP-01 (technical due diligence): Rust bumped 1.82 -> 1.88 to
# match `rust-version` in the root Cargo.toml. `--locked` added
# so container builds fail loudly on lockfile drift instead of
# silently picking up a different dependency set than CI/local.
# HEALTHCHECK rewired to a curl-based probe (the `--health-check`
# CLI flag does not exist; the previous form would have failed
# every probe and masked startup failures).
#
# Stage names match what `docker-compose.yml` references via
# `target:`:
#   - builder : compiles the binary
#   - runtime  : minimal Debian-slim image with the binary + curl
#
# Note: no `cargo install sccache` here. We rely on Docker layer
# caching for build performance; an sccache stage can be added
# later if build times justify the complexity.

# ---- builder ----
FROM rust:1.88-bookworm AS builder

WORKDIR /app

# Install system dependencies. `cmake` and `clang` are needed by
# some Candle build scripts; `protobuf-compiler` is needed by the
# `vllm-dist` gRPC build path.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    llvm \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better layer caching. We deliberately
# copy the toolchain pin so the container picks up the same MSRV
# as the host (matches `just ci`).
COPY rust-toolchain.toml ./
COPY Cargo.toml Cargo.lock ./
COPY crates/traits/Cargo.toml crates/traits/
COPY crates/dist/Cargo.toml crates/dist/
COPY crates/model/Cargo.toml crates/model/
COPY crates/core/Cargo.toml crates/core/
COPY crates/server/Cargo.toml crates/server/
COPY crates/testing/Cargo.toml crates/testing/

# Create dummy main.rs for dependency caching.
RUN mkdir -p crates/server/src && \
    echo "fn main() {}" > crates/server/src/main.rs

# Build dependencies only. `--locked` enforces the committed
# Cargo.lock; combined with the dummy main.rs this is the heavy
# dep-graph compile that we want to cache.
RUN cargo build --locked --release --bin vllm-server 2>/dev/null || true

# Copy actual source code.
COPY crates/traits/src crates/traits/src
COPY crates/dist/src crates/dist/src
COPY crates/model/src crates/model/src
COPY crates/core/src crates/core/src
COPY crates/testing/src crates/testing/src
COPY crates/server/src crates/server/src

# Touch main.rs to invalidate the dummy.
RUN touch crates/server/src/main.rs

# Build the actual application. `--locked` again — if this fails
# because the lockfile drifted during the COPY above, we want
# the container build to fail loudly, not silently update.
RUN cargo build --locked --release --bin vllm-server

# ---- runtime ----
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies. `curl` is used by HEALTHCHECK;
# `ca-certificates` is required for outbound HTTPS (e.g. weight
# download if the operator opts in).
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000 matches the Helm chart's
# `runAsUser` and the project's existing convention).
RUN useradd -m -u 1000 -s /bin/bash vllm

# Copy binary from builder.
COPY --from=builder /app/target/release/vllm-server /usr/local/bin/vllm-server

# Set ownership and switch to non-root.
RUN chown vllm:vllm /usr/local/bin/vllm-server
USER vllm

# Expose the HTTP API port (matches `service.port` in the Helm
# chart and the default `--port` in the CLI).
EXPOSE 8000

# Healthcheck: probe /health/live, which returns 200 once the
# HTTP server has bound the socket. We deliberately do not
# require the model to be loaded here — that is the readiness
# signal at /health/ready.
#
# DEP-01: previous form called `vllm-server --health-check`,
# which is not a real CLI flag. The container would have failed
# every probe and Docker would have cycled it indefinitely.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1

# Default args: bind 0.0.0.0:8000. `--model` is REQUIRED by the
# CLI; this default is the most permissive — operators are
# expected to override with `-e VLLM_MODEL=...` or a bind mount.
ENTRYPOINT ["vllm-server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
