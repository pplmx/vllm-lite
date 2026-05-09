# Testing

**Last updated:** 2026-05-09
**Focus:** Quality

## Test Framework

- **Runner**: `cargo nextest` (primary), `cargo test` (fallback)
- **Assertions**: Standard `assert!` / `assert_eq!` macros
- **Async**: `#[tokio::test]` for async tests
- **Property-based**: `proptest` (in vllm-testing dev-dependencies)
- **Benchmarks**: `criterion` 0.8

## Test Locations

### Inline Unit Tests (`#[cfg(test)]` modules)
~100 files contain inline `mod tests` blocks with unit tests co-located with implementation.

### Integration Test Files

**vllm-core** (`crates/core/tests/`):
- `scheduler.rs`, `scheduler_integration.rs` — Scheduler tests
- `integration.rs` — Core integration flows
- `e2e_concurrent.rs` — Concurrent request handling
- `e2e_lifecycle.rs` — Request lifecycle
- `e2e_request_lifecycle.rs` — Detailed request lifecycle
- `e2e_error_recovery.rs` — Error recovery scenarios
- `e2e_graceful_shutdown.rs` — Graceful shutdown
- `prefix_cache.rs` — Prefix caching
- `adaptive_speculative.rs` — Adaptive speculative decoding
- `beam.rs` — Beam search
- `error_handling.rs` — Error handling
- `engine_trace.rs` — Engine trace verification
- `observer.rs` — Scheduler observer
- `packing_integration.rs` — Token packing
- `resource_limits.rs` — Resource limit handling
- `sampling.rs` — Sampling strategies
- `cuda_graph_integration.rs` — CUDA Graph integration

**vllm-model** (`crates/model/tests/`):
- `model.rs` — Full model forward pass
- `attention.rs`, `attention_batch_benchmark.rs` — Attention tests
- `gqa_shape_tests.rs` — GQA shape verification
- `kv_cache_batch.rs` — KV cache batch operations
- `logits.rs` — Logits computation
- `checkpoint_loading_tests.rs` — Checkpoint loading (7 tests, all `#[ignore]`)
- `ssm_optimization_tests.rs` — SSM optimization
- `tiled_attention.rs` — Tiled attention
- `token_verification.rs` — Tokenizer verification

**vllm-server** (`crates/server/tests/`):
- `models_handler_test.rs` — Model listing endpoint

**vllm-traits** (`crates/traits/tests/`):
- `model_backend.rs` — ModelBackend trait tests

## Test Architecture

### Mock Model Pattern
Tests use `StubModel` (in `crates/core/src/engine.rs`) or `FakeModel` which implement `ModelBackend`:

```rust
#[derive(Clone)]
struct StubModel {
    token_to_return: TokenId,
}
impl ModelBackend for StubModel {
    fn forward(...) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| self.token_to_return).collect(),
        })
    }
}
```

### Test Utilities (`crates/testing/`)
The `vllm-testing` crate provides reusable test infrastructure:
- `harness.rs` — Test harness
- `request_factory.rs` — Request builder helpers
- `slow_model.rs` — Slow model simulation for timeout testing
- `builders/` — Builder patterns
- `fixtures/` — Test fixtures
- `mocks/` — Mock implementations
- `utils/` — Utility functions

## Test Execution

### Commands

```bash
# Fast tests only (skips #[ignore])
just nextest
# => cargo nextest run --workspace --all-features --no-fail-fast

# All tests including slow ones
just nextest-all
# => cargo nextest run --release --workspace --all-features --run-ignored all --no-fail-fast

# Single test
cargo test -p vllm-core test_engine_streaming
cargo test -p vllm-model -- attention

# Full CI (fmt + clippy + docs + all tests)
just ci
```

### Slow Tests
- Marked with `#[ignore]` attribute
- Located primarily in `crates/model/tests/checkpoint_loading_tests.rs` (7 tests)
- Skipped by default in `just nextest`, included in `just nextest-all`
- These tests require downloading model weights and are integration-level

## Coverage

- Tool: `cargo tarpaulin`
- Command: `just cov` (`cargo tarpaulin --all-features --workspace --exclude-files 'src/bin/*'`)
- Target: unspecified minimum (no coverage gates in CI)

## Benchmarks

- Framework: `criterion` 0.8
- Locations:
  - `crates/core/benches/` — Core benchmarks (scheduler, prefix cache, optimization)
  - `crates/model/benches/attention.rs` — Attention benchmarks
  - `benches/` — Top-level benchmarks (integration, attention, scheduler)

```bash
# Run all benchmarks
cargo bench --workspace --all-features --no-fail-fast

# Quick benchmarks (core only)
cargo bench -p vllm-core -p vllm-lite-benchmarks -- --test-threads=1
```

## Quality Gates (CI Pipeline)

```bash
# Full CI checklist
just ci  # runs: fmt-check, clippy, doc-check, nextest-all
```

The CI pipeline enforces:
1. `cargo fmt --all --check` — Formatting
2. `cargo clippy --all-targets --workspace -- -D warnings` — Lint (no warnings)
3. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items` — Documentation
4. `cargo nextest run --release --workspace --all-features --run-ignored all` — All tests

## Test Quality Observations

- **Strengths**: Extensive inline unit tests, comprehensive integration tests, dedicated testing crate, structured benchmark suite
- **Gaps**: No coverage threshold enforcement, checkpoint loading tests are all `#[ignore]` (no fast-path model verification), server integration tests minimal (only models handler test), no property-based tests in test suite
