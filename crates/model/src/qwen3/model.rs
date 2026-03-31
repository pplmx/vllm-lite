#![allow(clippy::all, unused)]
use crate::config::Qwen3Config;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};
use std::collections::HashMap;
use vllm_core::engine::ModelBackend;
use vllm_core::error::{EngineError, Result as EngineResult};
use vllm_core::types::{BatchOutput, BlockId, SeqId, TokenId};

use super::block::TransformerBlock;

pub struct Qwen3Model {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl Qwen3Model {
    pub fn new(config: Qwen3Config, device: Device) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers() {
            let layer = TransformerBlock::new(
                hidden_size,
                config.num_attention_heads(),
                config.num_key_value_heads(),
                hidden_size / config.num_attention_heads(),
                config.intermediate_size(),
                config.rms_norm_eps(),
                None,
            )?;
            layers.push(layer);
        }

        let norm = candle_nn::layer_norm(
            hidden_size,
            config.rms_norm_eps(),
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device),
        )?;
        let lm_head = candle_nn::linear(
            hidden_size,
            vocab_size,
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device),
        )?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            hidden_size / config.num_attention_heads(),
            1024,
            device.clone(),
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

    fn get_weight<'a>(weights: &'a HashMap<String, Tensor>, keys: &[&str]) -> Option<&'a Tensor> {
        for key in keys {
            if let Some(w) = weights.get(*key) {
                return Some(w);
            }
        }
        None
    }

    pub fn from_weights(
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
    ) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embed_key = "model.embed_tokens.weight";
        let embed_tokens = if let Some(w) = weights.get(embed_key) {
            Embedding::new(w.clone(), hidden_size)
        } else {
            return Err(candle_core::Error::msg(format!(
                "Missing weight: {}",
                embed_key
            )));
        };

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers() {
            let q_key = Self::get_weight(
                &weights,
                &[
                    &format!("model.layers.{}.self_attn.q_proj.weight", i),
                    &format!("model.layers.{}.attn.q_proj.weight", i),
                ],
            );
            let k_key = Self::get_weight(
                &weights,
                &[
                    &format!("model.layers.{}.self_attn.k_proj.weight", i),
                    &format!("model.layers.{}.attn.k_proj.weight", i),
                ],
            );
            let v_key = Self::get_weight(
                &weights,
                &[
                    &format!("model.layers.{}.self_attn.v_proj.weight", i),
                    &format!("model.layers.{}.attn.v_proj.weight", i),
                ],
            );
            let o_key = Self::get_weight(
                &weights,
                &[
                    &format!("model.layers.{}.self_attn.o_proj.weight", i),
                    &format!("model.layers.{}.attn.o_proj.weight", i),
                ],
            );

            let layer_weights = Some((
                q_key.cloned(),
                k_key.cloned(),
                v_key.cloned(),
                o_key.cloned(),
                weights
                    .get(&format!("model.layers.{}.mlp.gate_proj.weight", i))
                    .cloned(),
                weights
                    .get(&format!("model.layers.{}.mlp.up_proj.weight", i))
                    .cloned(),
                weights
                    .get(&format!("model.layers.{}.mlp.down_proj.weight", i))
                    .cloned(),
                weights
                    .get(&format!("model.layers.{}.input_layernorm.weight", i))
                    .cloned(),
                weights
                    .get(&format!(
                        "model.layers.{}.post_attention_layernorm.weight",
                        i
                    ))
                    .cloned(),
            ));

            let layer = TransformerBlock::new_with_weights(
                hidden_size,
                config.num_attention_heads(),
                config.num_key_value_heads(),
                hidden_size / config.num_attention_heads(),
                config.intermediate_size(),
                config.rms_norm_eps(),
                layer_weights,
            )?;
            layers.push(layer);
        }

        let norm_key = "model.norm.weight";
        let norm = if let Some(w) = weights.get(norm_key) {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), &device)?;
            LayerNorm::new(w.clone(), bias, config.rms_norm_eps())
        } else {
            return Err(candle_core::Error::msg(format!(
                "Missing weight: {}",
                norm_key
            )));
        };

        let lm_head = if let Some(w) = Self::get_weight(
            &weights,
            &["lm_head.weight", "output.weight", "model.lm_head.weight"],
        ) {
            Linear::new(w.clone(), None)
        } else if let Some(embed) = weights.get("model.embed_tokens.weight") {
            Linear::new(embed.clone(), None)
        } else {
            return Err(candle_core::Error::msg(
                "Missing lm_head weight (or tied embedding)".to_string(),
            ));
        };

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            hidden_size / config.num_attention_heads(),
            1024,
            device.clone(),
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
        is_prefill: bool,
    ) -> EngineResult<(Tensor, Tensor)> {
        if tokens.is_empty() {
            return Err(EngineError::ModelError("Empty tokens".to_string()));
        }

        let token_tensor = Tensor::new(tokens, &self.device)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;
        let hidden = self
            .embed_tokens
            .forward(&token_tensor)
            .map_err(|e| EngineError::ModelError(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;

        let mut hidden = hidden;

        if is_prefill {
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                hidden = layer
                    .forward_prefill(&hidden, &mut self.kv_cache, layer_idx, block_ids)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
            }
        } else {
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                hidden = layer
                    .forward_decode(
                        &hidden,
                        &self.kv_cache,
                        layer_idx,
                        block_ids,
                        num_computed_tokens,
                    )
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
            }
        }

        hidden = self
            .norm
            .forward(&hidden)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;
        let logits = self
            .lm_head
            .forward(&hidden)
            .map_err(|e| EngineError::ModelError(e.to_string()))?;

        Ok((logits, hidden))
    }

    pub fn forward_quantized_demo(
        &mut self,
        input_tokens: &[TokenId],
    ) -> EngineResult<(Tensor, Tensor)> {
        self.forward_with_cache(input_tokens, 0, &[0], true)
            .map_err(|e| EngineError::ModelError(e.to_string()))
    }
}

impl ModelBackend for Qwen3Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<BatchOutput> {
        let mut next_tokens = Vec::with_capacity(seq_ids.len());

        for (seq_idx, tokens) in input_tokens.iter().take(seq_ids.len()).enumerate() {
            if tokens.is_empty() {
                next_tokens.push(0);
                continue;
            }

            let hidden_states = {
                let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
                let embed = self
                    .embed_tokens
                    .forward(&token_tensor)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
                embed
                    .unsqueeze(0)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?
            };

            let mut hidden_states = hidden_states;

            for layer in &self.layers {
                hidden_states = layer
                    .forward(&hidden_states)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
            }

            hidden_states = self
                .norm
                .forward(&hidden_states)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let logits = self
                .lm_head
                .forward(&hidden_states)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            // logits shape: [batch=1, seq_len, vocab_size]
            // Get the last token's logits: [vocab_size]
            let batch_size = logits.dims()[0];
            let seq_len = logits.dims()[1];
            let last_logits = logits
                .get(batch_size - 1)
                .map_err(|e| EngineError::ModelError(e.to_string()))?
                .get(seq_len - 1)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let max_idx = last_logits
                .argmax(0)
                .map_err(|e| EngineError::ModelError(e.to_string()))?
                .to_scalar::<u32>()
                .unwrap_or(0);

            next_tokens.push(max_idx as TokenId);
        }

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }
}
