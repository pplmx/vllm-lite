# Code Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan.

**Goal:** Add documentation and tests for new modules

---

## Task 1: Add doc comments to modules

**Files:**

- Modify: `crates/model/src/llama/mod.rs`
- Modify: `crates/model/src/mistral/mod.rs`
- Modify: `crates/model/src/config/mod.rs`
- Modify: `crates/model/src/config/architecture.rs`
- Modify: `crates/model/src/config/model_config.rs`

- [ ] **Step 1: Add doc to llama/mod.rs**

```rust
//! Llama model architecture implementation.
//! 
//! This module provides the LlamaModel and LlamaBlock structs
//! that implement the ModelBackend trait for Llama-style transformers.
//!
//! # Architecture
//! - Uses RMSNorm (instead of LayerNorm)
//! - Uses SwiGLU MLP
//! - Supports Grouped-Query Attention (GQA)
//!
//! # Example
//! ```rust
//! use crate::config::ModelConfig;
//! use candle_core::Device;
//! 
//! let config = ModelConfig::llama_7b();
//! let model = LlamaModel::new(config, Device::Cpu, 1024).unwrap();
//! ```
```

- [ ] **Step 2: Add doc to mistral/mod.rs**

```rust
//! Mistral model architecture implementation.
//! 
//! This module provides the MistralModel and MistralBlock structs
//! that implement the ModelBackend trait with sliding window attention.
//!
//! # Differences from Llama
//! - Uses sliding window attention (4096 tokens by default)
//! - Uses Grouped-Query Attention with fewer KV heads
//! - Supports MistralSparseMoe block in Mixtral
```

- [ ] **Step 3: Add doc to config modules**

```rust
//! Model configuration types.
//! 
//! Provides unified configuration for multiple model architectures
//! including Llama, Mistral, and Qwen3.
//!
//! # Usage
//! ```rust
//! use crate::config::{ModelConfig, Architecture};
//! 
//! // Use predefined configs
//! let llama = ModelConfig::llama_7b();
//! let mistral = ModelConfig::mistral_7b();
//! 
//! // Or create from HuggingFace config.json
//! let config = ModelConfig::from_config_json(&value).unwrap();
//! ```
```

- [ ] **Step 4: Run fmt and commit**

---

## Task 2: Add LlamaBlock unit tests

**Files:**

- Modify: `crates/model/src/llama/block.rs`

- [ ] **Step 1: Add tests module**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_llama_block_forward_shape() {
        let config = ModelConfig::llama_7b();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }

    #[test]
    fn test_llama_block_single_token() {
        let config = ModelConfig::llama_7b();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((1, 1, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, 4096]);
    }

    #[test]
    fn test_llama_block_different_batch_sizes() {
        let config = ModelConfig::llama_7b();

        for batch_size in [1, 2, 4] {
            let block = LlamaBlock::new(&config, 0).unwrap();
            let input = Tensor::ones((batch_size, 5, 4096), DType::F32, &Device::Cpu).unwrap();
            let output = block.forward(&input).unwrap();
            assert_eq!(output.dims()[0], batch_size);
        }
    }
}
```

- [ ] **Step 2: Run tests and commit**

---

## Task 3: Add MistralBlock unit tests

**Files:**

- Modify: `crates/model/src/mistral/block.rs`

- [ ] **Step 1: Add tests module**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_mistral_block_forward_shape() {
        let config = ModelConfig::mistral_7b();
        let block = MistralBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }

    #[test]
    fn test_mistral_block_sliding_window_config() {
        let config = ModelConfig::mistral_7b();
        let block = MistralBlock::new(&config, 0).unwrap();

        // Verify sliding window is set
        assert_eq!(block.sliding_window, 4096);
    }
}
```

- [ ] **Step 2: Run tests and commit**

---

## Task 4: Final verification

- [ ] **Step 1: Run all tests**

```bash
cargo test --workspace
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Run fmt check**

```bash
cargo fmt --all --check
```

- [ ] **Step 4: Commit**

---

## Summary

- Task 1: Add doc comments
- Task 2: Add LlamaBlock tests
- Task 3: Add MistralBlock tests
- Task 4: Final verification
