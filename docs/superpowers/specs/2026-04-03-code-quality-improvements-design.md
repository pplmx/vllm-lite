# Code Quality Improvements Design

**Date**: 2026-04-03
**Status**: Approved
**Goal**: Add documentation and tests for new modules

## Current State

- Clippy: ✅ Clean
- Tests: ✅ 285 tests passing
- Format: ✅ Clean

**Missing:**

- Doc comments for new modules (llama, mistral, config)
- Unit tests for LlamaBlock and MistralBlock

## Target State

### 1. Module Documentation

Add doc comments to:

```rust
// crates/model/src/llama/mod.rs
//! Llama model architecture implementation.
//! 
//! This module provides the LlamaModel and LlamaBlock structs
//! that implement the ModelBackend trait for Llama-style transformers.
//!
//! # Example
//! ```rust
//! use crate::config::ModelConfig;
//! use candle_core::Device;
//! 
//! let config = ModelConfig::llama_7b();
//! let model = LlamaModel::new(config, Device::Cpu, 1024).unwrap();
//! ```

```rust
// crates/model/src/mistral/mod.rs
//! Mistral model architecture implementation.
//! 
//! This module provides the MistralModel and MistralBlock structs
//! that implement the ModelBackend trait with sliding window attention.
//!
//! # Differences from Llama
//! - Uses sliding window attention (4096 tokens)
//! - Different num_kv_heads configuration
```

```rust
// crates/model/src/config/mod.rs
//! Model configuration types.
//! 
//! Provides unified configuration for multiple model architectures
//! including Llama, Mistral, and Qwen3.
```

### 2. Unit Tests

Add tests for LlamaBlock and MistralBlock:

```rust
// crates/model/src/llama/block.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device};

    #[test]
    fn test_llama_block_forward_shape() {
        let config = ModelConfig::llama_7b();
        let block = LlamaBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }

    #[test]
    fn test_llama_block_rms_norm() {
        // Test RMS norm computation
    }

    #[test]
    fn test_llama_different_batch_sizes() {
        // Test with batch_size 1, 2, 4, 8
    }
}

// crates/model/src/mistral/block.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device};

    #[test]
    fn test_mistral_block_forward_shape() {
        let config = ModelConfig::mistral_7b();
        let block = MistralBlock::new(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }
}
```

## Implementation Order

1. Add doc comments to all new modules
2. Add LlamaBlock unit tests
3. Add MistralBlock unit tests
4. Run tests to verify

## Acceptance Criteria

- [ ] All new modules have doc comments
- [ ] LlamaBlock has at least 3 unit tests
- [ ] MistralBlock has at least 2 unit tests
- [ ] All tests pass
