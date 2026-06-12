#![allow(dead_code)]

use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::kv_cache::PagedKvCache;
use candle_core::{D, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result as EngineResult, SeqId, TokenId};

type EngineError = vllm_traits::ModelError;

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
            config.num_heads, // Use expanded num_heads for GQA
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

    pub fn from_weights(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", embed_key)))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            layers.push(LlamaBlock::from_weights(&config, i, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {}", norm_key)))?;
        let norm = Linear::new(norm_weight, None);

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_weight, None)
        } else {
            let lm_key = "lm_head.weight";
            let lm_weight = weights
                .get(lm_key)
                .cloned()
                .or_else(|| weights.get("model.embed_tokens.weight").cloned())
                .ok_or_else(|| candle_core::Error::msg("Missing lm_head.weight"))?;
            Linear::new(lm_weight, None)
        };

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads, // Use expanded num_heads for GQA
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

    pub fn forward_with_cache(
        &mut self,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> EngineResult<(Tensor, usize)> {
        if tokens.is_empty() {
            let logits = Tensor::zeros(
                (1, 1, self.config.vocab_size),
                candle_core::DType::F32,
                &self.device,
            )
            .map_err(|e| EngineError::new(e.to_string()))?;
            return Ok((logits, 0));
        }

        let hidden = if is_prefill {
            let t =
                Tensor::new(tokens, &self.device).map_err(|e| EngineError::new(e.to_string()))?;
            self.embed_tokens
                .forward(&t)
                .map_err(|e| EngineError::new(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| EngineError::new(e.to_string()))?
        } else {
            let last_token = tokens.last().copied().unwrap_or(0);
            let t = Tensor::new(&[last_token], &self.device)
                .map_err(|e| EngineError::new(e.to_string()))?;
            self.embed_tokens
                .forward(&t)
                .map_err(|e| EngineError::new(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| EngineError::new(e.to_string()))?
        };

        let mut hidden = hidden;
        if is_prefill {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer
                    .forward_prefill(&hidden, &mut self.kv_cache, layer_idx, block_ids, positions)
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }
        } else {
            let decode_position = [positions[0]];
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer
                    .forward_decode(
                        &hidden,
                        &mut self.kv_cache,
                        layer_idx,
                        block_ids,
                        num_computed_tokens,
                        &decode_position,
                    )
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }
        }

        hidden = self
            .norm
            .forward(&hidden)
            .map_err(|e| EngineError::new(e.to_string()))?;
        let logits = self
            .lm_head
            .forward(&hidden)
            .map_err(|e| EngineError::new(e.to_string()))?;

        Ok((logits, 0))
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
    ) -> EngineResult<BatchOutput> {
        if seq_ids.is_empty() {
            return Ok(BatchOutput {
                seq_ids: vec![],
                next_tokens: vec![],
            });
        }

        let mut next_tokens = vec![0u32; seq_ids.len()];
        for i in 0..seq_ids.len() {
            let (logits, _) = self.forward_with_cache(
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                is_prefill[i],
            )?;

            let next = if is_prefill[i] {
                let seq_len = logits.dims()[1];
                logits
                    .narrow(1, seq_len - 1, 1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .argmax(D::Minus1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec0::<u32>()
                    .map_err(|e| EngineError::new(e.to_string()))?
            } else {
                logits
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .argmax(D::Minus1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec0::<u32>()
                    .map_err(|e| EngineError::new(e.to_string()))?
            };
            next_tokens[i] = next;
        }

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
    ) -> EngineResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(seq_ids.len());
        for i in 0..seq_ids.len() {
            let (logits, _) = self.forward_with_cache(
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                is_prefill[i],
            )?;

            let logits_vec = if is_prefill[i] {
                let seq_len = logits.dims()[1];
                logits
                    .narrow(1, seq_len - 1, 1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| EngineError::new(e.to_string()))?
            } else {
                logits
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| EngineError::new(e.to_string()))?
            };
            results.push(logits_vec);
        }
        Ok(results)
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(input_tokens.len());
        for tokens in input_tokens {
            if tokens.is_empty() {
                results.push(vec![0.0; self.config.hidden_size]);
                continue;
            }
            let t = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| EngineError::new(e.to_string()))?;
            let emb = self
                .embed_tokens
                .forward(&t)
                .map_err(|e| EngineError::new(e.to_string()))?
                .mean(0)
                .map_err(|e| EngineError::new(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| EngineError::new(e.to_string()))?;
            results.push(emb);
        }
        Ok(results)
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
