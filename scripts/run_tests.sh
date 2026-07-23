#!/usr/bin/env bash
set -e
cd /workspace/vllm-lite
cargo test -p vllm-core --features opentelemetry 2>&1
