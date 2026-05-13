# Testing Patterns

**Analysis Date:** 2026-05-13

## Test Framework

**Runner:**

- Primary: `cargo nextest` (via `just nextest` / `just nextest-all`)
- Fallback: `cargo test` (used in CI for broad compatibility)
- CI command: `cargo test --workspace`

**Nextest Config:** `.config/nextest.toml`

- Three profiles: `default`, `ci`, `optimized`
- `fail-fast = false` everywhere (collect all failures)
- Retry with exponential backoff for flaky tests
- Slow test timeout: 30s default, 60s CI
- Test groups for concurrency control: `heavy-model-tests` (max 2 threads), `kernel-tests` (max 4 threads)

**Run Commands:**

```bash
# Nextest (skips #[ignore] tests)
just nextest

# Nextest including all tests including #[ignore]
just nextest-all

# Single crate tests
cargo test -p vllm-core

# Run specific test by name filter
cargo test -p vllm-core test_engine_streaming
cargo test -p vllm-model -- attention

# Run with output shown
cargo test --workspace -- --nocapture

# Run ignored/slow tests
cargo test -- --ignored
```

**Assertion Framework:**

- Standard Rust `assert!`, `assert_eq!`, `assert_ne!` — no external assertion library
- `unwrap()` used liberally in tests (no error propagation needed for panics)
- `std::panic::catch_unwind` used for testing expected panics (e.g., `crates/testing/src/utils/mod.rs:69`)

## Test File Organization

**Location:**

- **Unit tests:** Inline `#[cfg(test)] mod tests { ... }` at the bottom of each source file
- **Extracted test modules:** Separate `tests.rs` in same directory when tests grow large (e.g., `crates/core/src/scheduler/packing/tests.rs`, `crates/core/src/scheduler/policy/tests.rs`)
- **Integration tests:** Top-level `tests/` directories per crate:
    - `crates/core/tests/integration.rs` (1023 lines)
    - `crates/core/tests/speculative_kv_cache.rs`
    - `crates/core/tests/speculative_memory_overhead.rs`
    - `crates/core/tests/adaptive_speculative.rs`
    - `crates/model/tests/model.rs`
    - `crates/model/tests/token_verification.rs`
    - `crates/model/tests/checkpoint_loading_tests.rs`

**Naming:**

- Unit test modules: `mod tests` within the file being tested
- Unit test functions: `test_<component>_<expected_behavior>` (e.g., `test_engine_add_request`, `test_slow_model_forward_takes_time`)
- Integration test files: `tests/*.rs` with descriptive names (`integration.rs`, `speculative_kv_cache.rs`)

**Structure:**

```text
crates/core/
├── src/
│   ├── engine.rs              # Contains #[cfg(test)] mod tests
│   ├── scheduler/
│   │   ├── engine.rs          # Contains #[cfg(test)] mod tests
│   │   ├── packing/
│   │   │   ├── mod.rs
│   │   │   └── tests.rs       # Extracted test module
│   │   └── policy/
│   │       ├── mod.rs
│   │       └── tests.rs       # Extracted test module
│   └── error/
│       └── mod.rs             # Contains #[cfg(test)] mod tests
└── tests/
    ├── integration.rs         # Integration tests
    ├── speculative_kv_cache.rs
    └── adaptive_speculative.rs

crates/testing/                 # Dedicated test infrastructure crate
└── src/
    ├── lib.rs                 # Re-exports all test utilities
    ├── harness.rs             # TestHarness for env setup
    ├── mocks/
    │   └── mod.rs             # FakeModel, StubModel, etc.
    ├── request_factory.rs     # RequestFactory for generating requests
    ├── slow_model.rs          # SlowModel with configurable delay
    ├── fixtures/
    │   └── mod.rs             # TestFixtures with preset configs
    ├── builders/
    │   └── mod.rs             # RequestBuilder, BatchBuilder
    └── utils/
        └── mod.rs             # assert_batch_consistency, create_simple_batch
```

## Test Structure

**Suite Organization:**

```rust
// Pattern from crates/core/src/scheduler/engine.rs:634
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Request;
    use vllm_traits::BatchPhase;

    // Helper function for test setup
    fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        SchedulerEngine::new(config, num_kv_blocks, metrics)
    }

    #[test]
    fn test_engine_add_request() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(id > 0);
        assert!(engine.has_pending());
        assert_eq!(engine.waiting_count(), 1);
    }
}
```

**Setup Patterns:**

- Private helper functions for test fixtures: `create_test_engine()`, `make_sequence()`, `test_config()`
- `TestHarness` for integration-level scheduler tests (from `vllm_testing` crate)
- `RequestFactory` for generating test requests with configurable parameters
- `BatchBuilder` / `RequestBuilder` for constructing test data
- `TestFixtures` for preset configurations (small batch, chunked prefill, PD separation, OOM scenario)

**Teardown:**

- No explicit teardown — Rust's ownership model handles cleanup
- Tests that create resources are self-contained within function scope

**Assertion Patterns:**

```rust
// Simple assertions
assert!(id > 0);
assert_eq!(engine.waiting_count(), 1);

// Descriptive assertion messages
assert!(!batch.is_empty(), "Batch should contain at least one sequence");
assert!(
    batch.input_tokens[0].len() <= 10,
    "Chunk should respect target size"
);

// Float comparison with epsilon
assert!((metrics.cuda_graph_hit_rate() - 0.67).abs() < 0.01);

// Channel assertions
assert!(rx1.try_recv().is_ok(), "req1 should get token in step 1");

// Range checks
assert!((0.0..=1.0).contains(&pressure));
```

## Mocking

**Framework:** No mocking framework (no `mockall` or similar). The project uses **hand-rolled mock implementations** via the `ModelBackend` trait.

**Dedicated Mock Crate:** `vllm-testing` (`crates/testing/`) provides all mock models:

| Mock                 | File                                  | Behavior                                                 |
| -------------------- | ------------------------------------- | -------------------------------------------------------- |
| `StubModel`          | `crates/testing/src/mocks/mod.rs:10`  | Returns `seq_id` (converted) as next token               |
| `IncrementModel`     | `crates/testing/src/mocks/mod.rs:78`  | Returns `seq_id` as next token                           |
| `ConstModel`         | `crates/testing/src/mocks/mod.rs:147` | Always returns a fixed `return_token`                    |
| `FakeModel`          | `crates/testing/src/mocks/mod.rs:223` | Returns `seq_id % vocab_size` as token                   |
| `NeverProgressModel` | `crates/testing/src/mocks/mod.rs:304` | Always returns same token (for timeout/preemption tests) |
| `SlowModel`          | `crates/testing/src/slow_model.rs:25` | Configurable `Duration` delay on each forward            |

**Mock Pattern:**

```rust
// All mocks implement ModelBackend trait directly
pub struct StubModel;

impl ModelBackend for StubModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],          // Unused params prefixed with _
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| 1 as TokenId).collect(),
        })
    }
    // ... vocab_size(), num_layers(), num_heads() return hardcoded values
}
```

**Usage in Tests:**

```rust
// Integration test usage (from crates/core/tests/integration.rs:5)
use vllm_testing::{ConstModel, IncrementModel};

let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);
```

```rust
// Unit test usage (from crates/core/src/engine.rs:714)
struct StubModel {
    token_to_return: TokenId,
}
// Local impl ModelBackend for inline testing
```

**What to Mock:**

- The `ModelBackend` trait — always mock the model when testing scheduler/engine logic
- Use the dedicated `vllm-testing` crate for cross-crate test utilities
- Inline test-only structs when a custom token return value is needed

**What NOT to Mock:**

- Scheduler internals — test the real `SchedulerEngine` with mocked models
- Data structures (`Batch`, `Sequence`, `Request`) — test them directly
- Core types — use the real implementations

## Fixtures and Factories

**TestHarness (unified integration setup):**

```rust
// crates/testing/src/harness.rs
let harness = TestHarness::new()
    .kv_blocks(128)
    .max_batch_size(8)
    .build();

let mut scheduler = harness.scheduler();
let seq_id = scheduler.add_request(Request::new(0, vec![1, 2, 3], 10));
```

**RequestFactory (programmatic request generation):**

```rust
// crates/testing/src/request_factory.rs
let mut factory = RequestFactory::new()
    .min_tokens(64)
    .max_tokens(512)
    .max_max_tokens(32);

let request = factory.create();           // Single request
let requests = factory.create_batch(10);  // 10 requests
```

**TestFixtures (preset configurations):**

```rust
// crates/testing/src/fixtures/mod.rs
TestFixtures::default_scheduler_config()   // Standard 256-seq config
TestFixtures::small_batch_config()         // max_num_seqs=2
TestFixtures::chunked_prefill_config()     // token budget=10
TestFixtures::pd_separation_config()       // Prefill/Decode separation
TestFixtures::priority_config()            // Priority scheduling
TestFixtures::oom_scenario_config()        // Single sequence, tight budget
```

**Test Data Location:**

- Fixtures live in `crates/testing/src/fixtures/mod.rs`
- Factories live in `crates/testing/src/request_factory.rs`
- Builders live in `crates/testing/src/builders/mod.rs`
- Model test configs use `ModelConfig::test_tiny()` in `crates/model/src/config/model_config.rs`

## Coverage

**Tool:** `cargo-tarpaulin` (via `just cov`)

```bash
just cov    # Generates coverage report
```

Underlying command:

```bash
cargo tarpaulin --all-features --workspace --exclude-files 'src/bin/*'
```

**Requirements:** None enforced in CI — no coverage gates. Tarpaulin is available for local development.

## Test Types

**Unit Tests:**

- Located inline in `#[cfg(test)] mod tests` blocks
- Test individual functions, methods, and small components
- Use locally-defined helper functions for setup
- Examples: `test_seq_not_found_error_message`, `test_batch_respects_max_size`, `test_greedy_sample`

**Integration Tests:**

- Located in crate-level `tests/` directories
- Test end-to-end flows with `Engine` + mock models
- Use `vllm_testing` crate's mocks and fixtures
- Examples: `test_continuous_batching_with_streaming`, `test_chunked_prefill_integration`, `test_max_tokens_includes_prompt`

**Slow/Integration Tests (ignored by default):**

- Marked with `#[ignore]` attribute (sometimes with reason: `#[ignore = "slow integration test..."]`)
- 36 tests currently ignored across the workspace:
    - `crates/core/tests/` — speculative decoding, memory overhead tests
    - `crates/model/tests/` — full-size model tests, token verification, checkpoint loading
    - `crates/core/src/engine/speculative.rs` — speculative engine tests
- Run with: `just nextest-all` or `cargo test -- --ignored`

**E2E Tests:** Not used — no browser-based or HTTP-level end-to-end testing.

**Benchmarks:**

- Located in `benches/` workspace member
- Run via `just bench` (all) or `just bench-quick` (scheduler + core only)
- Uses `cargo bench` (Rust's built-in benchmark harness)

## Common Patterns

**Helper Functions for Test Data:**

```rust
// crates/core/src/scheduler/batch_composer.rs:406
fn make_sequence(id: u64, tokens: Vec<u32>, status: Status) -> Sequence {
    Sequence {
        id,
        tokens,
        kv_blocks: Arc::new(vec![id as usize]),
        num_computed_tokens: 0,
        prompt_len: 3,
        status,
        max_tokens: 10,
        sampling_params: SamplingParams::default(),
        consecutive_decode_rounds: 0,
        priority: Priority::default(),
    }
}
```

**Async Testing:**

```rust
// Tokio channels for streaming tests
use tokio::sync::mpsc;

let (tx, mut rx) = mpsc::channel(64);
engine.add_request(Request::new(1, vec![10, 20], 5), tx);
engine.step().unwrap();
assert!(rx.try_recv().is_ok());
```

**Error Testing:**

```rust
// crates/core/src/error/mod.rs:30
#[test]
fn test_seq_not_found_error_message() {
    let err = EngineError::SeqNotFound { id: 42 };
    assert_eq!(err.to_string(), "sequence 42 not found");
}
```

**Model Output Testing (candle tensors):**

```rust
// crates/model/src/kernels/flash_attention.rs:590
#[test]
fn test_scaled_dot_product_attention() -> Result<()> {
    let sdpa = ScaledDotProductAttention::new(64);
    let q = Tensor::ones((2, 8, 10, 64), DType::F32, &Device::Cpu)?;
    let k = Tensor::ones((2, 8, 10, 64), DType::F32, &Device::Cpu)?;
    let v = Tensor::ones((2, 8, 10, 64), DType::F32, &Device::Cpu)?;
    let output = sdpa.forward(&q, &k, &v)?;
    assert_eq!(output.dims(), &[2, 8, 10, 64]);
    Ok(())
}
```

**Builder Pattern Testing:**

```rust
// crates/testing/src/slow_model.rs:135
#[test]
fn test_slow_model_builder() {
    let model = SlowModel::new(Duration::from_secs(1)).return_token(100);
    assert_eq!(model.delay, Duration::from_secs(1));
    assert_eq!(model.return_token, 100);
}
```

## CI Test Pipeline

From `.github/workflows/ci.yml`:

| Job           | Command                                         | Purpose                       |
| ------------- | ----------------------------------------------- | ----------------------------- |
| `check`       | `cargo fmt --all -- --check`                    | Format validation             |
| `check`       | `cargo clippy --workspace -- -D warnings`       | Lint                          |
| `check`       | `cargo audit \|\| true`                         | Security audit (non-blocking) |
| `test`        | `cargo build --workspace --no-default-features` | Build check (CPU)             |
| `test`        | `cargo test --workspace`                        | Run all tests                 |
| `test`        | `cargo test --workspace -- --nocapture`         | Tests with full output        |
| `docs`        | `cargo doc --workspace --no-deps`               | Documentation build           |
| `matrix-test` | `cargo test --workspace` on stable + beta       | Rust version matrix           |
| `security`    | `cargo audit`                                   | Security audit                |

**Environment:**

- `RUST_BACKTRACE=1` for full backtraces
- `CANDLE_NATIVE_BACKEND=1` disables CUDA (GitHub runners lack GPU)
- Cache: `~/.cargo/` and `target/` cached between runs

## Test Count & Distribution

Approximate distribution across the workspace:

| Crate          | Unit Tests (#[test]) | Test files (src)          | Integration tests (tests/) |
| -------------- | -------------------- | ------------------------- | -------------------------- |
| `vllm-core`    | ~125                 | 42 `#[cfg(test)]` modules | 4 test files               |
| `vllm-model`   | ~376                 | 59 `#[cfg(test)]` modules | 3 test files               |
| `vllm-server`  | ~18                  | 18 `#[cfg(test)]` modules | 0                          |
| `vllm-testing` | ~18                  | 5 `#[cfg(test)]` modules  | 0                          |
| `vllm-traits`  | ~3                   | 1 `#[cfg(test)]` module   | 0                          |
| `vllm-dist`    | ~5                   | 2 `#[cfg(test)]` modules  | 0                          |

---

*Testing analysis: 2026-05-13*
