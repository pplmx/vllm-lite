#!/usr/bin/env sh
cd /workspace/vllm-lite
cargo clippy -p vllm-core --features opentelemetry --all-targets 2>&1
