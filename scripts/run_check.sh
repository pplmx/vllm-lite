#!/usr/bin/env bash
set -e
cargo test -p vllm-core --features opentelemetry --no-run 2>&1
