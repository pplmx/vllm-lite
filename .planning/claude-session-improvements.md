# Claude Session Improvements Log

## 2026-07-22 — Quality improvements

### Clippy warnings fixed
- `doc_markdown`: Wrapped identifier-like words in backticks in doc comments (RoPE, YaARN, rope_gqa, KV, etc.)
- `or_fun_call`: Changed `map_or` to `map_or_else` in `build_rope()` — `RoPE::new()` has side effects
- `redundant_clone`: Removed unnecessary `.clone()` on `device` in 4 test files (layer_loop, decoder_block/mod.rs, mixtral/block/tests.rs, tensor_store/mod.rs)
- `no_effect_underscore_binding`: Changed `let _device =` to `let _ =` in gqa/tests.rs (7 occurrences)
- `suboptimal_flops`: Used `mul_add()` in tensor_store test data
- `long_literal_lacking_separators`: Added `_` separators (1_000_000.0, 1_664_525, 1_013_904_223)
- `match_single_binding`: Converted `match msg { AddRequest { .. } => { ... } _ => {} }` to `if let` in test_fixtures.rs
- `missing_debug_implementations`: Added `#[derive(Debug)]` to 3 test structs in otlp_stub_collector.rs

### Bug fixes
- `{n}` interpolation in test messages: `expect("Some({n})...")` was literal text, not format interpolation. Fixed with `format!()` and `expect_err()`.
- Nested backtick issues in doc comments: `` `Arc<Mutex<`PagedKvCache`>>` `` → `` `Arc<Mutex<PagedKvCache>>` ``

### Files modified
- crates/model/src/components/positional/rope.rs
- crates/model/src/components/attention/gqa/forward.rs
- crates/model/src/components/attention/gqa/mod.rs
- crates/model/src/components/attention/rope_gqa.rs
- crates/model/src/causal_lm/hybrid_lm.rs
- crates/model/src/causal_lm/model/mod.rs
- crates/model/src/causal_lm/layer_loop.rs
- crates/model/src/config/model_config.rs
- crates/model/src/components/decoder_block/mod.rs
- crates/model/src/components/decoder_block/factory.rs
- crates/model/src/components/vision.rs
- crates/model/src/kv_cache.rs
- crates/model/src/loader/builder.rs
- crates/model/src/paged_tensor/tensor_store/mod.rs
- crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs
- crates/model/src/mixtral/block/tests.rs
- crates/model/src/mixtral/block/mod.rs
- crates/model/src/arch/stub.rs
- crates/model/src/qwen3/config/model/qwen3_config.rs
- crates/model/src/qwen3/config/model/text_config.rs
- crates/model/src/qwen3/config/rope.rs
- crates/server/src/test_fixtures.rs
- crates/server/src/openai/sampling_validation.rs
- crates/server/tests/otlp_stub_collector.rs

### Status
- All 1822 tests pass (40 skipped)
- `cargo fmt --check` passes
- `cargo clippy -D clippy::correctness -D clippy::suspicious -D clippy::perf` passes
- `cargo doc -- -D warnings` passes
- 35 commits ahead of origin/main