# Codebase Concerns

**Analysis Date:** 2026-04-26

## Technical Debt

### Qwen3.5 Hybrid Block Abstraction Incomplete

**Issue:** The `Architecture::create_block` method returns `todo!()` for Qwen3.5 hybrid architecture. Instead, the system uses model-level integration (`crates/model/src/qwen3_5/hybrid.rs`).

**Files:**
- `crates/model/src/qwen3_5/arch.rs:76`
- `crates/model/src/qwen3_5/hybrid.rs`

**Impact:** Cannot use the standard `TransformerBlock` abstraction for hybrid Mamba+Attention models. Changes to the hybrid model require modifying the monolithic hybrid module rather than using composable block patterns.

**Fix approach:** Implement `Qwen3Block` following the pattern of other model blocks (`LlamaBlock`, `MistralBlock`) to properly integrate with the architecture registry system.

---

### Test Architecture Has Unimplemented Methods

**Issue:** The test struct `TestArch` in the registry test module has `todo!()` in `create_block()` and `create_model()` methods, making it unsuitable for integration testing.

**Files:** `crates/model/src/arch/registry.rs:97,106`

**Impact:** The test architecture cannot be used for end-to-end architecture testing. Only unit tests exist for the registry.

**Fix approach:** Either implement stub methods or use a real model like `LlamaArchitecture` for integration tests.

---

## Known Bugs and Limitations

### Ignored Tests (7 tests)

**Issue:** Seven integration tests are marked `#[ignore]` in `crates/model/tests/checkpoint_loading_tests.rs` because they require external model files.

**Files:**
- `crates/model/tests/checkpoint_loading_tests.rs:69,124,167,189,215,269,299`

**Tests:**
- `test_qwen35_weight_keys` - Requires `/models/Qwen3.5-0.8B`
- `test_qwen35_remapped_weight_structure` - Requires `/models/Qwen3.5-0.8B`
- `test_qwen3_tokenizer_roundtrip` - Requires `/models/Qwen3-0.6B`
- `test_qwen3_direct_inference` - Requires `/models/Qwen3-0.6B`
- `test_qwen3_weight_diagnostics` - Requires `/models/Qwen3-0.6B`
- `test_qwen3_qk_norm_weights` - Requires `/models/Qwen3-0.6B`
- `test_all_models_loadable` - Requires multiple model directories

**Impact:** Critical checkpoint loading and multi-model loading workflows are not validated in CI.

**Fix approach:** Either download models during test setup or mock the checkpoint loading for unit tests.

---

## Unimplemented Features

### From ROADMAP.md (Long-term Vision)

| Feature | Status | Impact |
|---------|--------|--------|
| Pipeline Parallelism | Not started | Cannot scale across multiple GPUs with pipeline strategy |
| Distributed KV Cache | Not started | Memory cannot be shared across GPU nodes |
| Mobile/Edge Deployment | Not started | Cannot deploy to resource-constrained environments |
| WebAssembly Support | Not started | Cannot run in browsers or edge runtimes |
| Online Fine-tuning Interface | Not started | Cannot adapt models at runtime |

**Files:** `ROADMAP.md:110-111,255-260`

**Fix approach:** These are long-term roadmap items. Pipeline parallelism should be prioritized for multi-GPU deployments.

---

### Quantization Limitations

**Issue:** Only GGUF Q4_K_M quantization is supported. Other formats (GPTQ, AWQ, INT8) have stubs but no implementation.

**Files:**
- `crates/model/src/paged_tensor/quantization.rs`
- `crates/model/Cargo.toml:22`

**Current support:**
- FP16 (native)
- FP32 (native)
- GGUF Q4_K_M (dequantizes to FP16)

**Missing:**
- GPTQ support
- AWQ support
- INT8 Weight-Only quantization runtime
- INT8 KV Cache runtime

**Fix approach:** Implement quantization kernels for each format following the candle quantization patterns.

---

## Performance Bottlenecks

### Hash-based Prefix Cache (Collision Risk)

**Issue:** The prefix cache uses a simple u64 hash for cache keys. The hash function in `crates/core/src/kv_cache/prefix_cache.rs:83-92` converts floats to u64 via truncation, which can cause hash collisions.

```rust
// Current hash - collisions likely
data.iter().map(|&x| (x.abs() * 1000.0) as u64)
    .fold(0u64, |acc, x| acc.wrapping_mul(31).wrapping_add(x))
```

**Files:** `crates/core/src/kv_cache/prefix_cache.rs:83-92`

**Impact:** Hash collisions can cause incorrect prefix matches, returning wrong cached blocks for prompts.

**Fix approach:** Use a cryptographic hash (e.g., xxhash, sha256) with proper byte serialization.

---

### Speculative Decoding Fallback

**Issue:** When speculative decoding is enabled but no draft model is set, the system falls back silently without logging a performance warning.

**Files:** `crates/core/src/engine/speculative.rs:60-64`

**Impact:** Users may enable speculative decoding expecting acceleration but get no speedup without realizing the draft model isn't loaded.

**Fix approach:** Return an error during initialization if speculative decoding is enabled without a draft model, rather than falling back silently.

---

## Architecture Issues

### Heavy Use of expect/unwrap (1162 occurrences)

**Issue:** The codebase has 1162 instances of `.expect()`, `.unwrap()`, and `.unwrap_or()` that can panic on unexpected conditions.

**Files:** Throughout codebase, concentrated in:
- `crates/model/tests/*.rs`
- `crates/model/src/loader/*.rs`
- `crates/server/src/openai/chat.rs:263,283`

**Impact:** Unexpected errors (malformed models, corrupted data) cause panics rather than graceful error handling.

**Fix approach:** Replace with proper `Result` propagation and user-friendly error messages.

---

### Model Loading Panics

**Issue:** The main server binary uses `panic!()` for model loading failures, causing process termination instead of graceful degradation.

**Files:** `crates/server/src/main.rs:117,121,135`

```rust
.unwrap_or_else(|e| panic!("Failed to create loader: {}", e));
.unwrap_or_else(|e| panic!("Failed to load model: {}", e));
.unwrap_or_else(|e| panic!("Failed to load draft model: {}", e));
```

**Impact:** A corrupted model file crashes the entire server instead of returning an error to the user.

**Fix approach:** Return a proper error code with descriptive message instead of panicking.

---

## Security Considerations

### API Key Authentication Location

**Issue:** The `crates/server/src/auth/` directory was not found. Auth middleware is referenced in main.rs but the implementation is unclear.

**Files:**
- `crates/server/src/main.rs:41` (references `Option<Arc<AuthMiddleware>>`)
- `crates/server/src/auth/mod.rs` (missing)

**Impact:** API key authentication may not be properly implemented or may be a placeholder.

**Fix approach:** Verify auth middleware implementation exists and add integration tests.

---

### TLS/SSL Not Implemented

**Issue:** Per ROADMAP.md, TLS/SSL is delegated to external components (nginx).

**Files:** `ROADMAP.md:227`

**Impact:** API traffic is unencrypted by default. Production deployments require external termination.

**Mitigation:** Document that production deployments must use reverse proxy with TLS.

---

### Unsafe Code (4 instances)

**Issue:** Unsafe code is used for memory-mapped I/O and tensor data conversion.

**Files:** `crates/model/src/loader/io.rs:28,75,85,95`

```rust
// Line 28: mmap
unsafe { Mmap::map(&file) }

// Lines 75, 85, 95: slice conversion from raw pointers
unsafe { std::slice::from_raw_parts(tensor_data.as_ptr() as *const u16, n) }
```

**Impact:** Memory-mapped files and raw pointer conversions can lead to undefined behavior if bounds are incorrect or file is modified during access.

**Mitigation:** The mmap threshold checks and length calculations are conservative. Audit bounds carefully if modifying tensor loading.

---

### Unsafe Send Implementation

**Issue:** `CudaGraph` uses `unsafe impl Send` with a comment claiming thread safety.

**Files:** `crates/model/src/kernels/cuda_graph.rs:47`

```rust
// SAFETY: CudaGraph can be Send because it only contains thread-safe types
unsafe impl Send for CudaGraph {}
```

**Impact:** If assumptions about thread safety are incorrect, data races may occur.

**Fix approach:** Verify that `Arc<dyn CudaGraphNode>` and all contained types are truly `Send + Sync`. Add integration tests with concurrent execution.

---

## Fragile Areas

### Qwen3.5 Weight Key Remapping

**Issue:** The weight key remapping function for Qwen3.5 is brittle and depends on specific naming conventions.

**Files:** `crates/model/src/qwen3_5/arch.rs:15-65`

**Why fragile:** Any change to HuggingFace model naming conventions will break model loading.

**Safe modification:** Add validation that expected keys exist before remapping, with clear error messages.

---

### Prefix Cache Hash Invalidation

**Issue:** The prefix match cache is cleared on every insert, potentially losing valid prefix matches.

**Files:** `crates/core/src/kv_cache/prefix_cache.rs:93`

```rust
self.prefix_match_cache.clear();
```

**Impact:** Repeated prefix cache lookups for non-cached prefixes may perform unnecessary searches.

**Fix approach:** Implement LRU eviction for `prefix_match_cache` instead of full clear.

---

## Scaling Limits

### Block Size Fixed at Compile Time

**Issue:** `BLOCK_SIZE` is defined as a constant in `vllm_traits::BLOCK_SIZE`.

**Files:** `crates/traits/src/types.rs`

**Impact:** Cannot tune block size for different hardware configurations without recompilation.

**Fix approach:** Make block size configurable via CLI/env at startup.

---

### Single-device CUDA Support

**Issue:** CUDA device selection is limited to single device (device 0 or CPU fallback).

**Files:** `crates/server/src/main.rs:106`

```rust
let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
```

**Impact:** Cannot utilize multiple GPUs for single-request parallelism.

**Fix approach:** Add multi-GPU configuration and tensor parallelism support.

---

## Dependencies at Risk

### candle-core 0.10.2

**Issue:** The candle version is pinned. Newer candle versions may have breaking API changes.

**Files:**
- `crates/model/Cargo.toml:11-12`
- `Cargo.toml:workspace.dependencies`

**Risk:** Upgrading candle may require significant changes to tensor operations.

**Fix approach:** Pin to minor version and test upgrades in isolation.

---

### gguf 0.1

**Issue:** GGUF support uses a relatively new crate (version 0.1).

**Files:** `crates/model/Cargo.toml:22`

**Risk:** Breaking changes in gguf crate updates.

**Fix approach:** Review gguf changelog before updating, maintain integration tests.

---

## Test Coverage Gaps

### No Integration Tests for Auth Middleware

**What's not tested:** API key authentication middleware has no dedicated tests.

**Files:** `crates/server/src/auth/` (missing)

**Risk:** Auth bypass vulnerabilities could go undetected.

**Priority:** High

---

### No E2E Tests with Real Models

**What's not tested:** End-to-end inference with real HuggingFace models.

**Files:** All `#[ignore]` tests in `crates/model/tests/checkpoint_loading_tests.rs`

**Risk:** Model loading bugs may only appear with real model files.

**Priority:** Medium

---

### No Concurrency Tests for Scheduler

**What's not tested:** Concurrent request handling in `SchedulerEngine`.

**Files:** `crates/core/src/scheduler/engine.rs`

**Risk:** Data races in request queue or memory manager could cause corruption.

**Priority:** High

---

*Concerns audit: 2026-04-26*
