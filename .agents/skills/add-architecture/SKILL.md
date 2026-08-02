---
name: add-architecture
description: >
  Step-by-step workflow for adding a new model architecture to vllm-lite.
  Use when the user wants to add support for a new model family (e.g., Phi,
  Gemma, new Qwen variant), create a new Architecture trait implementation,
  register it in the architecture registry, or asks "how do I add a new model".
  Covers the full flow: module scaffold, Architecture impl, CausalLm/HybridLm
  model type, block definition, register.rs, register_all_archs, and verification.
---

# Add a New Model Architecture

## Key insight: use `CausalLm` / `HybridLm`, not raw `ModelBackend`

All production architectures delegate `ModelBackend` to a generic shell:

| Shell      | Use when                                   | Example                                |
| ---------- | ------------------------------------------ | -------------------------------------- |
| `CausalLm` | Standard decoder-only transformer          | Llama, Mistral, Qwen3, Gemma4, Mixtral |
| `HybridLm` | Mixed layer types (attention + linear/SSM) | Qwen3.5                                |

The model type is a **type alias**, not a struct:

```rust
pub type LlamaModel = CausalLm<LlamaBlock, RmsNorm, Linear>;
pub type MistralModel = CausalLm<MistralBlock, LnLayerNorm, Linear>;
pub type Qwen35HybridModel = HybridLm<HybridBlock, LayerNorm, Qwen3Config>;
```

## Three block patterns

**Pattern A — Re-export** (Llama, Mistral): block uses standard RoPE+GQA+SwiGLU.

```rust
// block.rs — entire file
pub use crate::components::decoder_block::{
    RopeGqaDecoderBlock as LlamaBlock, block_from_weights, new_block,
};
```

**Pattern B — Custom block** (Mixtral MoE, Gemma4 hybrid attention): implement
`PagedDecoderBlock` trait with specialized attention/MLP, plus `new()` and
`block_from_weights()` constructors.

**Pattern C — Hybrid blocks** (Qwen3.5): multiple block types behind an enum,
used with `HybridLm`.

## Step-by-step

### 1. Scaffold

Create `crates/model/src/<name>/` and declare in `lib.rs`:

```text
crates/model/src/<name>/
├── mod.rs        # Docs + re-exports
├── arch.rs       # Architecture trait impl
├── block.rs      # Block type (re-export or custom PagedDecoderBlock)
├── model.rs      # type alias + constructors
└── register.rs   # Registry factory
```

### 2. `block.rs`

**If standard RoPE+GQA+SwiGLU** (Pattern A):

```rust
pub use crate::components::decoder_block::{
    RopeGqaDecoderBlock as <Name>Block, block_from_weights, new_block,
};
```

**If custom attention/MLP** (Pattern B): implement `PagedDecoderBlock`:

```rust
use crate::components::decoder_block::PagedDecoderBlock;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};

#[derive(Debug)]
pub struct <Name>Block { /* attention, mlp, norms */ }

impl PagedDecoderBlock for <Name>Block {
    fn forward_prefill(&self, x: &Tensor, kv: &mut PagedKvCache,
        layer_idx: usize, block_ids: &[usize], positions: &[usize]) -> Result<Tensor> { ... }
    fn forward_prefill_continue(&self, x: &Tensor, kv: &mut PagedKvCache,
        layer_idx: usize, block_ids: &[usize], positions: &[usize],
        num_computed: usize) -> Result<Tensor> { ... }
    fn forward_decode(&self, x: &Tensor, kv: &mut PagedKvCache,
        layer_idx: usize, block_ids: &[usize], num_computed: usize,
        positions: &[usize]) -> Result<Tensor> { ... }
}
```

Provide `pub fn new_block(config, layer_idx, device) -> Result<<Name>Block>`
and `pub fn block_from_weights(config, layer_idx, weights) -> Result<<Name>Block>`.

Reference: `crates/model/src/mixtral/block.rs` (MoE), `crates/model/src/gemma4/block.rs`.

### 3. `model.rs`

```rust
use crate::causal_lm::CausalLm;
use crate::components::RmsNorm;  // or LnLayerNorm
use crate::config::ModelConfig;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Linear;
use std::collections::HashMap;
use super::block::{block_from_weights, new_block};

pub type <Name>Block = super::block::<Name>Block;
pub type <Name>Model = CausalLm<<Name>Block, RmsNorm, Linear>;

impl <Name>Model {
    /// # Errors
    /// Returns `Err` if tensor allocation or weight loading fails.
    pub fn new(config: ModelConfig, device: &Device, num_kv_blocks: usize) -> CandleResult<Self> {
        Self::new_rms(config, device.clone(), num_kv_blocks, false, |c, idx| {
            new_block(c, idx, device)
        })
    }

    /// # Errors
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_weights(
        config: ModelConfig, device: &Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize, kv_quantization: bool,
    ) -> CandleResult<Self> {
        Self::from_hf_weights_rms(config, device.clone(), weights,
            num_kv_blocks, kv_quantization, block_from_weights)
    }
}
```

Use `new_rms` / `from_hf_weights_rms` for RMSNorm models (Llama, Gemma4).
Use `new_with_block_fn` / `from_hf_weights_ln` for LayerNorm models (Mistral, Mixtral, Qwen3).

### 4. `arch.rs`

```rust
use crate::arch::{ArchCapabilities, Architecture};
use crate::causal_lm::BlockWrapper;
use crate::components::TransformerBlock;
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result, Tensor};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use vllm_traits::ModelBackend;

#[derive(Debug)]
pub struct <Name>Architecture;

impl <Name>Architecture {
    #[must_use]
    pub const fn new() -> Self { Self }
}

impl Default for <Name>Architecture {
    fn default() -> Self { Self::new() }
}

impl Architecture for <Name>Architecture {
    fn name(&self) -> &'static str { "<name>" }

    fn detect(&self, config_json: &serde_json::Value) -> bool {
        let model_type = config_json.get("model_type")
            .and_then(|v| v.as_str()).unwrap_or("");
        matches!(model_type.to_lowercase().as_str(), "<name>")
    }

    fn capabilities(&self) -> ArchCapabilities {
        ArchCapabilities::PRODUCTION
    }

    fn create_block(&self, config: &ModelConfig, layer_idx: usize,
        weights: &HashMap<String, Tensor>, _device: &Device,
    ) -> Result<Box<dyn TransformerBlock>> {
        let block = super::block::block_from_weights(config, layer_idx, weights)?;
        Ok(Box::new(BlockWrapper::new(block, config)))
    }

    fn create_model(&self, config: ModelConfig, device: Device,
        weights: HashMap<String, Tensor>, num_kv_blocks: usize, kv_quantization: bool,
    ) -> Result<(Box<dyn ModelBackend>, Option<Arc<Mutex<PagedKvCache>>>)> {
        let model = super::model::<Name>Model::from_weights(
            config, &device, weights, num_kv_blocks, kv_quantization)?;
        let kv_cache = model.paged_kv_cache();
        Ok((Box::new(model), Some(kv_cache)))
    }
}
```

### 5. `register.rs`

```rust
use std::sync::Arc;
use crate::arch::{Architecture, ArchitectureRegistry};
use super::arch::<Name>Architecture;

pub fn register(registry: &ArchitectureRegistry) {
    let factory: Arc<dyn Fn() -> Box<dyn Architecture> + Send + Sync> =
        Arc::new(|| Box::new(<Name>Architecture::new()));
    registry.register("<name>", factory);
}
```

### 6. Wire up

- `mod.rs`: re-export `<Name>Architecture`, `<Name>Block`, `<Name>Model`
- `lib.rs`: add `pub mod <name>;`
- `arch/registry.rs`: add `crate::<name>::register::register(registry);` to `register_all_archs()`

### 7. Add test config

Add a preset in `config/model_config.rs` or use `ModelConfig::test_tiny_for(Architecture::...)`.

### 8. Verify

```bash
cargo build -p vllm-model
cargo test -p vllm-model -- <name>
just clippy && cargo fmt --all --check
```

## Checklist

- [ ] `block.rs` — re-export or custom `PagedDecoderBlock`
- [ ] `model.rs` — type alias to `CausalLm`/`HybridLm` + constructors
- [ ] `arch.rs` — `Architecture` trait (name, detect, capabilities, create_block, create_model)
- [ ] `register.rs` — factory function
- [ ] `mod.rs` + `lib.rs` + `register_all_archs()` wiring
- [ ] Unit tests with `ModelConfig::test_tiny()` or `test_tiny_for()`
- [ ] Clippy + fmt pass
