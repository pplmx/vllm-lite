---
name: add-component
description: >
  Workflow for adding or extending shared ML components in vllm-lite's
  components layer (crates/model/src/components/). Use when the user wants
  to add a new attention variant, normalization layer, MLP variant, positional
  encoding, SSM layer, or any reusable transformer building block. Also use
  when refactoring model-specific code into shared components or creating
  a new PagedDecoderBlock variant.
---

# Add a Shared ML Component

Components live in `crates/model/src/components/` and are composed by
architecture block implementations.

## Directory layout

```text
components/
├── attention/       # GQA (gqa/), MLA (mla.rs), paged (paged_gqa.rs), RoPE-GQA (rope_gqa.rs), flash
├── mlp/             # SwiGLU (swiglu.rs)
├── norm/            # RMSNorm (rms_norm.rs), LayerNorm (layer_norm.rs)
├── positional/      # RoPE (rope.rs), MRoPE (mrope.rs)
├── ssm/             # Mamba SSM, HarmonicSSM
├── gated_delta/     # Gated-delta linear attention
├── kv_cache_fp8/    # FP8 KV-cache quantization
├── decoder_block/   # RopeGqaDecoderBlock + PagedDecoderBlock trait + factory
├── block.rs         # TransformerBlock trait (extends PagedDecoderBlock)
├── vision.rs        # Vision encoder placeholder
└── mod.rs           # Re-exports
```

## Trait hierarchy

```text
PagedDecoderBlock          (forward_prefill, forward_prefill_continue, forward_decode)
  └── TransformerBlock     (+ inner_dim, num_kv_heads) — used by Architecture::create_block
```

Most architectures use `RopeGqaDecoderBlock` (standard RoPE+GQA+SwiGLU).
Custom blocks (Mixtral MoE, Gemma4) implement `PagedDecoderBlock` directly.

## Adding a new component

### 1. Create the module

Single-file: `components/<name>.rs`
Multi-file: `components/<name>/mod.rs` + submodules (add `tests.rs` for large test suites)

### 2. Follow the pattern

```rust
//! <Component description>.

use candle_core::{Result, Tensor};
use std::collections::HashMap;

/// Configuration for <Name>.
#[derive(Debug, Clone)]
pub struct <Name>Config {
    pub hidden_size: usize,
    // ...
}

/// <Name> layer implementation.
#[derive(Debug)]
pub struct <Name> {
    config: <Name>Config,
    // candle_nn layers: Linear, Embedding, etc.
}

impl <Name> {
    /// Create with random weights (for testing / new-block construction).
    /// # Errors
    /// Returns `Err` if tensor allocation fails.
    pub fn new(config: &<Name>Config, device: &candle_core::Device) -> Result<Self> { ... }

    /// Create from pre-loaded checkpoint weights.
    /// # Errors
    /// Returns `Err` if weight shapes don't match config.
    pub fn from_weights(config: &<Name>Config, weights: &HashMap<String, Tensor>) -> Result<Self> { ... }

    /// Forward pass.
    /// # Errors
    /// Returns `Err` on tensor operation failure.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { ... }
}
```

### 3. Register in `components/mod.rs`

```rust
/// <Description>.
pub mod <name>;
pub use <name>::{<Name>, <Name>Config};
```

### 4. Conventions

- Use `candle_core::Tensor` for all tensor operations
- Weight loading: accept `&HashMap<String, Tensor>` (key = checkpoint param name)
- Return `candle_core::Result<T>` from fallible functions
- Add `/// # Errors` doc on all `Result`-returning pub fns (deny-tier lint)
- `#[derive(Debug)]` on all public types (warn-tier lint)
- Single-letter vars (`q`, `k`, `v`, `b`, `h`, `d`) allowed in tensor math only
- Test with small shapes: `hidden_size=8, num_heads=2, seq_len=4`

### 5. Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn tiny_config() -> <Name>Config {
        <Name>Config { hidden_size: 8, /* ... */ }
    }

    #[test]
    fn test_<name>_forward_shape() {
        let config = tiny_config();
        let layer = <Name>::new(&config, &Device::Cpu).unwrap();
        let x = Tensor::ones((1, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8]);
    }

    #[test]
    fn test_<name>_from_weights() {
        let config = tiny_config();
        // Build weight map matching checkpoint key names
        let mut weights = HashMap::new();
        weights.insert("weight".into(),
            Tensor::ones((8, 8), DType::F32, &Device::Cpu).unwrap());
        let layer = <Name>::from_weights(&config, &weights).unwrap();
        let x = Tensor::ones((1, 4, 8), DType::F32, &Device::Cpu).unwrap();
        assert!(layer.forward(&x).is_ok());
    }
}
```

### 6. Verify

```bash
cargo test -p vllm-model -- <name>
cargo clippy -p vllm-model --all-features -- -D clippy::correctness -D clippy::suspicious -D clippy::perf
```

## Refactoring model code into a component

When extracting from an architecture (e.g., `gemma4/attention.rs` → `components/attention/`):

1. Move implementation to `components/<category>/<name>.rs`
2. Make it generic over config (accept `&AttentionConfig`, not model-specific config)
3. Re-export from `components/mod.rs`
4. Update the architecture to import from `crate::components`
5. Ensure existing architecture tests still pass: `cargo test -p vllm-model`
