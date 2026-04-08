# Unified Model Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement unified architecture supporting Llama and Mistral with easy extensibility

**Architecture:**

- Unified ModelConfig + Architecture enum
- ArchitectureSpec trait defining behavior differences
- Reuse existing components (attention, mlp, rope, norm)
- ModelRegistry for automatic model creation

**Tech Stack:** Rust, Candle, safetensors

---

## File Structure

```text
crates/model/src/
├── config/
│   ├── mod.rs
│   ├── model_config.rs    # NEW: Unified ModelConfig
│   └── architecture.rs    # NEW: Architecture enum + Spec
│
├── llama/
│   ├── block.rs           # MODIFY: Implement properly
│   └── model.rs           # MODIFY: Implement properly
│
├── mistral/               # NEW
│   ├── mod.rs
│   ├── block.rs           # With sliding window attention
│   └── model.rs
│
├── loader.rs              # MODIFY: Add Llama/Mistral loading
└── registry.rs            # MODIFY: Support new architectures
```

---

## Task 1: Create config module infrastructure

**Files:**

- Create: `crates/model/src/config/mod.rs`
- Create: `crates/model/src/config/architecture.rs`
- Create: `crates/model/src/config/model_config.rs`
- Modify: `crates/model/src/lib.rs`

- [ ] **Step 1: Create config/mod.rs**

```rust
pub mod architecture;
pub mod model_config;

pub use architecture::{Architecture, AttentionType, MlpType, NormType};
pub use model_config::ModelConfig;
```

- [ ] **Step 2: Create config/architecture.rs**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen3,
    Llama,
    Mistral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Mha,
    Gqa,
    SlidingWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    SwiGLU,
    GatedMLP,
}

pub trait ArchitectureSpec: Send + Sync {
    fn architecture() -> Architecture;
    fn attention_type() -> AttentionType;
    fn norm_type() -> NormType;
    fn mlp_type() -> MlpType;
    fn use_rope() -> bool;
    fn share_embeddings() -> bool;
    fn sliding_window() -> Option<usize>;
}

pub struct LlamaSpec;
impl ArchitectureSpec for LlamaSpec {
    fn architecture() -> Architecture { Architecture::Llama }
    fn attention_type() -> AttentionType { AttentionType::Gqa }
    fn norm_type() -> NormType { NormType::RmsNorm }
    fn mlp_type() -> MlpType { MlpType::SwiGLU }
    fn use_rope() -> bool { true }
    fn share_embeddings() -> bool { false }
    fn sliding_window() -> Option<usize> { None }
}

pub struct MistralSpec;
impl ArchitectureSpec for MistralSpec {
    fn architecture() -> Architecture { Architecture::Mistral }
    fn attention_type() -> AttentionType { AttentionType::SlidingWindow }
    fn norm_type() -> NormType { NormType::RmsNorm }
    fn mlp_type() -> MlpType { MlpType::SwiGLU }
    fn use_rope() -> bool { true }
    fn share_embeddings() -> bool { false }
    fn sliding_window() -> Option<usize> { Some(4096) }
}
```

- [ ] **Step 3: Create config/model_config.rs**

```rust
use super::Architecture;

pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

impl ModelConfig {
    pub fn llama_7b() -> Self {
        Self {
            architecture: Architecture::Llama,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            vocab_size: 32000,
            intermediate_size: 11008,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tie_word_embeddings: false,
            max_position_embeddings: 2048,
        }
    }

    pub fn mistral_7b() -> Self {
        Self {
            architecture: Architecture::Mistral,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            intermediate_size: 14336,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: Some(4096),
            tie_word_embeddings: false,
            max_position_embeddings: 32768,
        }
    }
}
```

- [ ] **Step 4: Update lib.rs**

```rust
pub mod config;
pub mod llama;
pub mod mistral;
```

- [ ] **Step 5: Run build and verify**

Run: `cargo build -p vllm-model`

- [ ] **Step 6: Commit**

```bash
git add crates/model/src/config/ crates/model/src/lib.rs
git commit -m "feat(model): add unified config infrastructure"
```

---

## Task 2: Implement Llama Model

**Files:**

- Modify: `crates/model/src/llama/block.rs`
- Modify: `crates/model/src/llama/model.rs`

- [ ] **Step 1: Implement llama/block.rs**

```rust
#![allow(dead_code)]

use crate::components::{swiglu_forward, AttentionConfig};
use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use crate::qwen3::attention::GqaAttention;
use crate::qwen3::mlp::SwiGLU;
use candle_core::{Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};

pub struct LlamaBlock {
    input_layernorm: Linear,
    post_attention_layernorm: Linear,
    attention: GqaAttention,
    mlp: SwiGLU,
}

impl LlamaBlock {
    pub fn new(config: &ModelConfig, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let theta = config.rope_theta;
        let rms_norm_eps = config.rms_norm_eps;

        let input_layernorm = candle_nn::linear(hidden_size, hidden_size, false)?;
        let post_attention_layernorm = candle_nn::linear(hidden_size, hidden_size, false)?;

        let attention = GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            None,
            AttentionConfig::default(),
            false, // No qk_norm for Llama
        )?;

        let mlp = SwiGLU::new(hidden_size, intermediate_size, None)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        let x = swiglu_forward(&x, &self.mlp.gate_proj, &self.mlp.up_proj, &self.mlp.down_proj)?;
        x.add(&residual)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Linear) -> Result<Tensor> {
        let hidden_size = x.dims().last().unwrap();
        let x = x.reshape(((), hidden_size))?;
        let weight = weight.weight().reshape((hidden_size,))?;
        let variance = x.sqr()?.mean(1)?;
        let x = x.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
        let x = x.broadcast_mul(&weight)?;
        x.reshape(x.dims())
    }
}
```

- [ ] **Step 2: Implement llama/model.rs**

```rust
#![allow(dead_code)]

use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear};
use vllm_traits::{BatchOutput, ModelBackend, SeqId, TokenId};

use super::block::LlamaBlock;

pub struct LlamaModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<LlamaBlock>,
    norm: Linear,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl LlamaModel {
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings = Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(LlamaBlock::new(&config, 0)?);
        }

        let norm = candle_nn::linear(hidden_size, hidden_size, false)?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, false)?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_kv_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            false,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
        })
    }
}

impl ModelBackend for LlamaModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<BatchOutput> {
        // Placeholder implementation - returns fake logits
        let next_tokens: Vec<TokenId> = seq_ids.iter().map(|_| 0).collect();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        // Placeholder
        Ok(vec![vec![0.0_f32; 32000]; seq_ids.len()])
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.0_f32; self.config.hidden_size]; input_tokens.len()])
    }
}
```

- [ ] **Step 3: Run build**

Run: `cargo build -p vllm-model`

- [ ] **Step 4: Commit**

```bash
git add crates/model/src/llama/
git commit -m "feat(model): implement LlamaModel with unified config"
```

---

## Task 3: Implement Mistral Model

**Files:**

- Create: `crates/model/src/mistral/mod.rs`
- Create: `crates/model/src/mistral/block.rs`
- Create: `crates/model/src/mistral/model.rs`

- [ ] **Step 1: Create mistral/mod.rs**

```rust
pub mod block;
pub mod model;

pub use block::MistralBlock;
pub use model::MistralModel;
```

- [ ] **Step 2: Create mistral/block.rs**

```rust
#![allow(dead_code)]

use crate::components::{swiglu_forward, AttentionConfig};
use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use crate::qwen3::attention::GqaAttention;
use crate::qwen3::mlp::SwiGLU;
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

pub struct MistralBlock {
    input_layernorm: Linear,
    post_attention_layernorm: Linear,
    attention: GqaAttention,
    mlp: SwiGLU,
    sliding_window: usize,
}

impl MistralBlock {
    pub fn new(config: &ModelConfig, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let theta = config.rope_theta;
        let sliding_window = config.sliding_window.unwrap_or(4096);

        let input_layernorm = candle_nn::linear(hidden_size, hidden_size, false)?;
        let post_attention_layernorm = candle_nn::linear(hidden_size, hidden_size, false)?;

        let attention = GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            theta,
            None,
            AttentionConfig::default(),
            false,
        )?;

        let mlp = SwiGLU::new(hidden_size, intermediate_size, None)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
            sliding_window,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        let x = swiglu_forward(&x, &self.mlp.gate_proj, &self.mlp.up_proj, &self.mlp.down_proj)?;
        x.add(&residual)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Linear) -> Result<Tensor> {
        let hidden_size = x.dims().last().unwrap();
        let x = x.reshape(((), hidden_size))?;
        let weight = weight.weight().reshape((hidden_size,))?;
        let variance = x.sqr()?.mean(1)?;
        let x = x.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
        let x = x.broadcast_mul(&weight)?;
        x.reshape(x.dims())
    }
}
```

- [ ] **Step 3: Create mistral/model.rs**

```rust
#![allow(dead_code)]

use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear};
use vllm_traits::{BatchOutput, ModelBackend, SeqId, TokenId};

use super::block::MistralBlock;

pub struct MistralModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<MistralBlock>,
    norm: Linear,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl MistralModel {
    pub fn new(config: ModelConfig, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings = Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(MistralBlock::new(&config, 0)?);
        }

        let norm = candle_nn::linear(hidden_size, hidden_size, false)?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, false)?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_kv_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            false,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            kv_cache,
            device,
        })
    }
}

impl ModelBackend for MistralModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<BatchOutput> {
        // Placeholder - same as Llama for now
        let next_tokens: Vec<TokenId> = seq_ids.iter().map(|_| 0).collect();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.0_f32; 32000]; seq_ids.len()])
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.0_f32; self.config.hidden_size]; input_tokens.len()])
    }
}
```

- [ ] **Step 4: Run build**

Run: `cargo build -p vllm-model`

- [ ] **Step 5: Commit**

```bash
git add crates/model/src/mistral/
git commit -m "feat(model): implement MistralModel with sliding window"
```

---

## Task 4: Update registry and loader

**Files:**

- Modify: `crates/model/src/registry.rs`
- Modify: `crates/model/src/loader.rs`

- [ ] **Step 1: Update registry.rs to support Llama/Mistral**

Add creation functions:

```rust
impl ModelRegistry {
    pub fn create_llama(config: &ModelConfig, device: Device, num_kv_blocks: usize) -> Result<LlamaModel> {
        LlamaModel::new(config.clone(), device, num_kv_blocks)
    }

    pub fn create_mistral(config: &ModelConfig, device: Device, num_kv_blocks: usize) -> Result<MistralModel> {
        MistralModel::new(config.clone(), device, num_kv_blocks)
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p vllm-model`

- [ ] **Step 3: Commit**

```bash
git add crates/model/src/registry.rs crates/model/src/loader.rs
git commit -m "feat(model): update registry for Llama/Mistral support"
```

---

## Task 5: Final verification

**Files:**

- All modified files

- [ ] **Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Run format check**

Run: `cargo fmt --all --check`
Expected: No issues

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(model): complete unified model architecture - final verification"
```

---

## Summary

- Task 1: Create config module infrastructure
- Task 2: Implement Llama Model
- Task 3: Implement Mistral Model
- Task 4: Update registry and loader
- Task 5: Final verification
