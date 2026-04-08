# Additional Tests Design - Missing Key Coverage

**Date**: 2026-04-03  
**Status**: Draft  
**Goal**: Add comprehensive tests for missing critical coverage areas

---

## 1. Executive Summary

This document outlines test additions for critical gaps identified in the current test suite. The focus is on areas with no current coverage that are essential for production reliability.

---

## 2. Missing Test Coverage by Priority

### High Priority

| Category        | Module             | Missing Tests                 |
| --------------- | ------------------ | ----------------------------- |
| **vllm-traits** | ModelBackend trait | No interface tests            |
| **vllm-core**   | Error handling     | No error propagation tests    |
| **vllm-core**   | Resource limits    | Memory pressure, OOM handling |
| **vllm-model**  | Model loader       | Weight loading from files     |

### Medium Priority

| Category       | Module               | Missing Tests      |
| -------------- | -------------------- | ------------------ |
| **vllm-core**  | CUDA Graph           | Capture/replay     |
| **vllm-core**  | Speculative Decoding | Draft verification |
| **vllm-model** | Quantization         | AWQ/GPTQ           |
| **vllm-model** | MoE                  | Expert routing     |

### Lower Priority

| Category      | Module          | Missing Tests         |
| ------------- | --------------- | --------------------- |
| **vllm-dist** | Tensor Parallel | Multi-GPU integration |

---

## 3. Detailed Test Requirements

### 3.1 vllm-traits Tests (HIGH PRIORITY)

**Why**: No tests validate the core trait interfaces

**Tests to add**:

```rust
// crates/traits/tests/model_backend.rs

#[test]
fn test_model_backend_forward_signature() {
    // Verify ModelBackend trait method signatures
}

#[test]
fn test_sampling_params_validation() {
    // Verify temperature, top_p, etc. are validated
}

#[test]
fn test_request_creation() {
    // Verify Request::new works correctly
}
```

### 3.2 Error Handling Tests (HIGH PRIORITY)

**Why**: No tests for error propagation from engine to API

**Tests to add**:

```rust
// crates/core/tests/error_handling.rs

#[test]
fn test_engine_error_propagates() {
    // When model forward fails, error should propagate
}

#[test]
fn test_invalid_request_rejected() {
    // Invalid prompts should be rejected early
}

#[tokio::test]
async fn test_channel_error_recovery() {
    // When channel fails, engine should handle gracefully
}
```

### 3.3 Resource Limit Tests (HIGH PRIORITY)

**Why**: Critical for production stability

**Tests to add**:

```rust
// crates/core/tests/resource_limits.rs

#[test]
fn test_kv_cache_oom_handling() {
    // When KV cache is full, should evict or reject
}

#[test]
fn test_memory_pressure_handling() {
    // Under memory pressure, should reduce batch size
}

#[test]
fn test_concurrent_request_limits() {
    // Max concurrent requests should be enforced
}
```

### 3.4 Model Loader Tests (HIGH PRIORITY)

**Why**: No tests for loading weights from disk

**Tests to add**:

```rust
// crates/model/tests/loader.rs

#[test]
fn test_load_config_from_json() {
    // Load config.json
}

#[test]
fn test_load_safetensors_weights() {
    // Load .safetensors files
}

#[test]
fn test_load_sharded_model() {
    // Load sharded model (model-00001-of-00002.safetensors)
}

#[test]
fn test_missing_weights_error() {
    // Missing expected weights should error
}
```

---

## 4. Implementation Plan

### Phase 1: Core Interfaces (vllm-traits)

- Add trait interface tests
- Verify trait method signatures
- Test trait object safety

### Phase 2: Error Handling

- Engine error propagation
- Channel error recovery
- Invalid request handling

### Phase 3: Resource Management

- KV cache OOM
- Memory pressure
- Request limits

### Phase 4: Model Loading

- Config loading
- Weight loading
- Sharded model loading

### Phase 5: Advanced Features

- CUDA Graph (optional - needs GPU)
- Speculative decoding (optional)
- Quantization (optional)

---

## 5. Success Criteria

- [ ] vllm-traits has tests
- [ ] Error handling tested
- [ ] Resource limits tested
- [ ] Model loader tested
- [ ] All tests pass

---

## 6. Test Execution

All new tests should:

- Be in appropriate test files (`tests/` directory or inline `#[cfg(test)]`)
- Follow naming convention: `test_<module>_<scenario>`
- Have clear assertions
- Be independent (no order dependency)
