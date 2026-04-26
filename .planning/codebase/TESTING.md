# Testing Patterns

**Analysis Date:** 2026-04-26

## Test Framework

**Test Runner:** `cargo test` with `cargo-nextest` for CI

**Configuration:**
- `justfile` for common test commands
- `#[ignore]` attribute for slow/integration tests
- Feature flags: `--all-features` for comprehensive testing

**Run Commands:**

```bash
# Run all tests (skips #[ignore] by default)
just nextest
cargo nextest run --workspace --all-features --no-fail-fast

# Run all tests including slow/ignored ones
just nextest-all
cargo nextest run --release --workspace --all-features --run-ignored all --no-fail-fast

# Run single test
cargo test -p vllm-core test_engine_streaming
cargo test -p vllm-model -- attention

# Coverage report
just cov
cargo tarpaulin --all-features --workspace --exclude-files 'src/bin/*'
```

## Test Organization

### Inline Unit Tests (`#[cfg(test)]` modules)

Located in the same file as implementation, at the end of the file:

```rust
// From `crates/core/src/sampling.rs`
pub fn greedy_sample(logits: &[f32]) -> TokenId {
    // ... implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_selects_max() {
        assert_eq!(greedy_sample(&[0.1, 0.5, 0.3]), 1);
    }

    #[test]
    fn test_greedy_first_on_tie() {
        assert_eq!(greedy_sample(&[0.5, 0.5, 0.3]), 0);
    }
}
```

**Key patterns:**
- `#[cfg(test)]` module at end of file
- `use super::*` to bring implementation into scope
- Tests co-located with the code they test

### Integration Tests (`tests/` directory)

Located in `tests/` subdirectory of each crate:

```
crates/
├── core/tests/
│   ├── integration.rs
│   ├── scheduler.rs
│   ├── error_handling.rs
│   ├── prefix_cache.rs
│   └── ...
├── model/tests/
│   ├── checkpoint_loading_tests.rs
│   ├── attention.rs
│   └── ...
└── traits/tests/
    └── model_backend.rs
```

**Example integration test:**
```rust
// From `crates/core/tests/integration.rs`
use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::types::{Request, SchedulerConfig};
use vllm_testing::{ConstModel, IncrementModel};

#[test]
fn test_continuous_batching_with_streaming() {
    let config = SchedulerConfig {
        max_num_seqs: 2,
        max_num_batched_tokens: 100,
        ..Default::default()
    };
    let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

    let (tx1, mut rx1) = mpsc::channel(64);
    let (tx2, mut rx2) = mpsc::channel(64);

    engine.add_request(Request::new(1, vec![10, 20], 4), tx1);
    engine.add_request(Request::new(2, vec![30, 40, 50], 5), tx2);

    engine.step().unwrap();
    assert!(rx1.try_recv().is_ok(), "req1 should get token in step 1");
    // ...
}
```

## Test Utilities (Mock Models)

Located in `crates/testing/src/mocks/mod.rs`:

| Model | Behavior | Use Case |
|-------|----------|----------|
| `StubModel` | Returns `seq_id` as next token | Prefix cache tests |
| `IncrementModel` | Returns `seq_id + 1` as next token | Integration tests |
| `ConstModel` | Returns constant token | Speculative decoding tests |
| `FakeModel` | Returns `seq_id % vocab_size` | Model tests |
| `NeverProgressModel` | Returns same token always | Timeout/preemption tests |

**Usage:**
```rust
use vllm_testing::{ConstModel, IncrementModel};

// In integration tests
let mut engine = Engine::with_config(IncrementModel, None, config, 4, 1024);

// For speculative decoding
let model = ConstModel::new(42);
let mut engine = Engine::new(model.clone(), Some(model));
engine.enable_speculative();
```

## Test Fixtures

Located in `crates/testing/src/fixtures/mod.rs`:

```rust
pub struct TestFixtures;

impl TestFixtures {
    pub fn default_scheduler_config() -> SchedulerConfig {
        SchedulerConfig::default()
    }

    pub fn small_batch_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 2,
            max_num_batched_tokens: 100,
            // ...
        }
    }

    pub fn oom_scenario_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 1,
            max_num_batched_tokens: 1,
            // ...
        }
    }
}
```

## Common Test Patterns

### Configuration Validation Tests

```rust
#[test]
#[should_panic(expected = "max_num_seqs must be > 0")]
fn test_scheduler_config_rejects_zero_max_seqs() {
    let _ = SchedulerConfig::new(
        0, // should panic
        100,
        10,
        // ...
    );
}
```

### Async Engine Tests

```rust
use tokio::sync::mpsc;

#[test]
fn test_engine_lifecycle() {
    let engine = Engine::new(IncrementModel, None);
    let (tx, _rx) = mpsc::channel(64);
    engine.add_request(Request::new(1, vec![10, 20], 5), tx);
    // ...
}
```

### Assertion Patterns

```rust
// Standard assertions
assert!(condition, "message if failed");
assert_eq!(expected, actual);
assert_ne!(expected, actual);

// With panic messages
assert!(
    steps <= 5,
    "should finish within expected steps, got {}",
    steps
);

// try_recv pattern
assert!(rx.try_recv().is_ok(), "should get token");
```

### Slow/Ignored Tests

```rust
// Mark slow tests with #[ignore]
#[test]
#[ignore]
fn test_model_inference_real() {
    // Test requiring actual model weights
}

// Run ignored tests with:
cargo test -- --ignored
just nextest-all
```

### Feature-Gated Tests

```rust
#[cfg(feature = "gguf")]
#[test]
fn test_gguf_loader_can_load() {
    use vllm_model::loader::format::{FormatLoader, GgufLoader};
    let path = Path::new("model.gguf");
    assert!(GgufLoader::can_load(path));
}
```

## Testing Best Practices

### Test Naming

Format: `test_<function>_<expected_behavior>`

```rust
#[test]
fn test_greedy_selects_max() { ... }

#[test]
fn test_scheduler_add_request_returns_valid_id() { ... }

#[test]
fn test_prefix_cache_hit_directly_decoding() { ... }
```

### Test Structure

1. **Setup** - Create engine/config with test fixtures
2. **Act** - Call the function being tested
3. **Assert** - Verify expected outcomes
4. **Cleanup** - Drop channels, clean up resources

### What to Mock

- Use `IncrementModel`, `ConstModel`, `FakeModel` for model backend
- Use `TestFixtures` for common configurations
- Don't mock internal scheduler components (test at integration level)

### What NOT to Mock

- Public API of `Engine`, `SchedulerEngine`
- Request/response channels
- Configuration validation

## CI Integration

**Full CI check:**
```bash
just ci  # fmt-check, clippy, doc-check, nextest-all
```

**Quick verification:**
```bash
just quick  # fix, doc-check, nextest
```

**Pre-commit checklist:**
```bash
cargo fmt --all
cargo clippy --all-targets --workspace -- -D warnings
cargo test --workspace
```

---

*Testing analysis: 2026-04-26*
