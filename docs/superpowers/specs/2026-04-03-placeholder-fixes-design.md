# Placeholder/Fake Implementation Fixes Design

**Date**: 2026-04-03  
**Status**: Draft  
**Issue**: Fake implementations and placeholder code found during code review

---

## 1. Executive Summary

This document outlines fixes for two placeholder/fake implementations identified during code review:
1. Embeddings endpoint returning zero vectors on failure
2. Qwen3.5 Mamba model marked as simplified placeholder

---

## 2. Issue 1: Embeddings Fallback Returns Zero Vectors

### Current Problem

**File**: `crates/server/src/openai/embeddings.rs:43-49`

```rust
let embeddings = match rx.recv().await {
    Some(emb) => emb,
    None => {
        let embedding_dim = 1024;
        req.input.iter().map(|_| vec![0.0; embedding_dim]).collect()
    }
};
```

**Issue**: When engine message fails, returns zero vectors instead of propagating error. This causes:
- Silent failures that are hard to debug
- Clients receive invalid data without knowing
- Wrong embeddings affect downstream tasks

### Proposed Solution

**Approach**: Return explicit error when embeddings computation fails

```rust
let embeddings = rx.recv().await
    .ok_or_else(|| (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse::new(
            "Failed to get embeddings from engine",
            "internal_error"
        ))
    ))?;
```

---

## 3. Issue 2: Qwen3.5 Mamba Implementation

### Current Problem

**File**: `crates/model/src/qwen3_5/model.rs:85`

```rust
eprintln!("Warning: Qwen3.5 Mamba implementation is simplified (placeholder)");
```

The Qwen3.5 Mamba model is not fully implemented - only a placeholder exists.

### Proposed Solution

**Recommendation**: Complete Mamba implementation

**Scope**:
1. Implement Mamba state space model architecture
2. Add proper SSM (State Space Model) layer support
3. Integrate with existing KV cache system
4. Remove placeholder warning

**Architecture**:
- Mamba uses selective state space models (SSM)
- Different from transformer attention mechanism
- Requires specialized forward pass

---

## 4. Implementation Plan

### Phase 1: Fix Embeddings Error Handling

1. Modify `embeddings.rs` to return error on failure
2. Add test for error case
3. Run tests

### Phase 2: Complete Mamba Implementation

1. Research Mamba SSM architecture requirements
2. Implement S6 (Mamba) layer
3. Integrate with Qwen3.5 model
4. Add tests
5. Remove placeholder warning

---

## 5. Success Criteria

- [ ] Embeddings endpoint returns proper error on engine failure
- [ ] No placeholder warnings in production code
- [ ] Qwen3.5 Mamba implementation is complete
- [ ] All tests pass