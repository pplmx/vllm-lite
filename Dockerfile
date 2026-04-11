# Multi-stage Dockerfile for vLLM-lite
# Build stage
FROM rust:1.82-bookworm AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    clang \
    llvm \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/traits/Cargo.toml crates/traits/
COPY crates/dist/Cargo.toml crates/dist/
COPY crates/model/Cargo.toml crates/model/
COPY crates/core/Cargo.toml crates/core/
COPY crates/server/Cargo.toml crates/server/
COPY crates/testing/Cargo.toml crates/testing/

# Create dummy main.rs for dependency caching
RUN mkdir -p crates/server/src && \
    echo "fn main() {}" > crates/server/src/main.rs

# Build dependencies
RUN cargo build --release --bin vllm-server 2>/dev/null || true

# Copy actual source code
COPY crates/traits/src crates/traits/src
COPY crates/dist/src crates/dist/src
COPY crates/model/src crates/model/src
COPY crates/core/src crates/core/src
COPY crates/testing/src crates/testing/src
COPY crates/server/src crates/server/src

# Touch main.rs to invalidate cache
RUN touch crates/server/src/main.rs

# Build the actual application
RUN cargo build --release --bin vllm-server

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash vllm

# Copy binary from builder
COPY --from=builder /app/target/release/vllm-server /usr/local/bin/vllm-server

# Set ownership
RUN chown vllm:vllm /usr/local/bin/vllm-server

# Switch to non-root user
USER vllm

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD vllm-server --health-check || exit 1

# Run the server
ENTRYPOINT ["vllm-server"]
CMD ["--port", "8000", "--host", "0.0.0.0"]
