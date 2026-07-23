#!/usr/bin/env bash
cd /workspace/vllm-lite
git add crates/core/src/tracing_init.rs crates/core/src/lib.rs crates/core/src/metrics/exporter/mod.rs crates/core/Cargo.toml Cargo.lock
echo "staged successfully"
git status --short
