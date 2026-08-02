---
name: write-test
description: >
  Write tests for vllm-lite using the project's testing infrastructure
  (vllm-testing crate: TestHarness, StubModel, RequestFactory, BatchBuilder).
  Use when the user wants to add unit tests, integration tests, e2e tests,
  or asks "how do I test this", "write a test for", "add test coverage".
  Covers mock models, test harness setup, request factories, and per-crate
  test conventions.
---

# Write Tests

## Test infrastructure (`vllm-testing` crate)

### Mock models

```rust
use vllm_testing::mocks::StubModel;

// Always returns token 1
let model = StubModel::default();

// Always returns a specific token
let model = StubModel::returning(42);
```

Also available: `ConstModel`, `NeverProgressModel` (for timeout/preemption tests).

### TestHarness

```rust
use vllm_testing::{TestHarness, TestHarnessConfig};

let harness = TestHarness::new(
    TestHarnessConfig::default()
        .kv_blocks(256)
        .max_batch_size(16)
        .enable_prefix_cache(true)
);

let seq_id = harness.add_request(vec![1, 2, 3], 10);
let output = harness.run_to_completion(seq_id);
```

### RequestFactory and BatchBuilder

```rust
use vllm_testing::RequestFactory;
use vllm_testing::builders::BatchBuilder;

let factory = RequestFactory::new();
let request = factory.create(vec![10, 20, 30], 50);

let batch = BatchBuilder::new()
    .with_seq_id(1)
    .with_tokens(vec![10, 20, 30])
    .build();
```

## Test placement

| Type              | Location                                    | Discovery    |
| ----------------- | ------------------------------------------- | ------------ |
| Unit tests        | `#[cfg(test)] mod tests {}` in source file  | `cargo test` |
| Integration tests | `crates/<crate>/tests/<topic>.rs`           | `cargo test` |
| Cross-crate e2e   | `crates/core/tests/e2e_*.rs`                | `cargo test` |
| Slow/GPU tests    | Mark with `#[ignore = "reason"]`            | `--ignored`  |

**Never** place `.rs` test files in `src/` outside `mod tests` blocks.

## Unit test pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_<function>_<expected_behavior>() {
        // Arrange
        let config = SchedulerConfig::default();
        let mut engine = SchedulerEngine::default();

        // Act
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 10));

        // Assert
        assert!(id > 0);
    }
}
```

## Integration test pattern

```rust
// crates/core/tests/<topic>.rs
use vllm_core::{SchedulerEngine, Request, SchedulerConfig};
use vllm_testing::mocks::StubModel;

#[test]
fn test_<scenario>_<expected>() {
    let mut engine = SchedulerEngine::default();
    let model = StubModel::returning(42);

    let seq_id = engine.add_request(Request::new(0, vec![1, 2, 3], 10));
    let batch = engine.build_batch();
    let output = model.forward(
        &batch.seq_ids, &batch.input_tokens, &batch.positions,
        &batch.kv_block_ids, &batch.num_computed_tokens, &batch.is_prefill,
    ).unwrap();

    assert_eq!(output.tokens[0], 42);
}
```

## Model-level test pattern

Use `ModelConfig::test_tiny()` for fast model tests:

```rust
#[cfg(test)]
mod tests {
    use crate::config::ModelConfig;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_<arch>_block_forward_shape() {
        let config = ModelConfig::test_tiny();
        let block = super::new_block(&config, 0, &Device::Cpu).unwrap();
        let input = Tensor::ones(
            (1, 4, config.hidden_size), DType::F32, &Device::Cpu,
        ).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, config.hidden_size]);
    }

    #[test]
    #[ignore = "slow integration test"]
    fn test_<arch>_full_size() {
        let config = ModelConfig::llama_7b();
        // ...
    }
}
```

## Naming convention

`test_<function_or_component>_<expected_behavior>`

Examples:

- `test_scheduler_add_request`
- `test_prefix_cache_hit_on_shared_prompt`
- `test_llama_block_forward_shape`
- `test_sampling_top_k_filters_correctly`

## Running tests

```bash
# All (fast, skips #[ignore])
just nextest

# Single crate
cargo test -p vllm-core

# By name
cargo test -p vllm-core -- test_scheduler

# With output
cargo test -p vllm-core -- test_name --nocapture

# Including slow tests
just nextest-all
```

## Conventions

- Use `FakeModel` / `StubModel` for mocking — never real model weights in unit tests
- Small tensor shapes: `hidden_size=8, num_heads=2, seq_len=4`
- `#[ignore = "reason"]` for slow/GPU tests with explanation string
- `proptest` for property-based tests (workspace dependency)
- Add `dev-dependencies` on `vllm-testing` in the crate's `Cargo.toml`
