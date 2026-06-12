//! Gemma4 Model implementation.
//!
//! Weight loading and forward pass are not implemented yet; registration exists so
//! configs detect correctly and tests can exercise block-level code.

use crate::config::ModelConfig;
use crate::gemma4::block::Gemma4Block;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear};
use std::collections::HashMap;
use vllm_traits::{BatchOutput, ModelBackend, SeqId, TokenId};

/// Weights are loaded in `from_weights`; `forward` is still a stub.
#[allow(dead_code)]
pub struct Gemma4Model {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<Gemma4Block>,
    norm: Linear,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl Gemma4Model {
    pub fn new(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        // Use a dummy VarBuilder for now
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(Gemma4Block::new(&config, i, vb.clone())?);
        }

        let norm_weight =
            Tensor::zeros((hidden_size, hidden_size), candle_core::DType::F32, &device)?;
        let norm = Linear::new(norm_weight, None);

        let lm_head_weight =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let lm_head = Linear::new(lm_head_weight, None);

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_kv_heads,
            config.head_dim,
            num_kv_blocks,
            device.clone(),
            kv_quantization,
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

    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        _weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
    ) -> CandleResult<Self> {
        Self::new(config, device, num_kv_blocks, kv_quantization)
    }
}

impl ModelBackend for Gemma4Model {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<BatchOutput> {
        let next_tokens: Vec<TokenId> = seq_ids.iter().map(|_| 0).collect();
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }

    fn forward_logits(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![0.0_f32; self.config.vocab_size]; seq_ids.len()])
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> vllm_traits::Result<Vec<Vec<f32>>> {
        Ok(vec![
            vec![0.0_f32; self.config.hidden_size];
            input_tokens.len()
        ])
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }
}
