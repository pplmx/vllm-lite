#![allow(dead_code)]

use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
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

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(LlamaBlock::new(&config, 0)?);
        }

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let norm = candle_nn::linear(hidden_size, hidden_size, vb.pp("norm"))?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

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
}
