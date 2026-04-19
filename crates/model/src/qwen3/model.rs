#![allow(dead_code)]
use crate::kv_cache::PagedKvCache;
use crate::qwen3_config::Qwen3Config;
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};
use std::collections::HashMap;
use vllm_dist::TensorParallelConfig;
use vllm_traits::{BatchOutput, BlockId, SeqId, TokenId};
use vllm_traits::{ModelBackend, Result as EngineResult};

pub type EngineError = vllm_traits::ModelError;

use super::block::TransformerBlock;

pub struct Qwen3Model {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
    #[allow(dead_code)]
    tp_config: Option<TensorParallelConfig>,
}

impl Qwen3Model {
    pub fn new(config: Qwen3Config, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        let theta = config.rope_theta();
        for _ in 0..config.num_hidden_layers() {
            let layer = TransformerBlock::new(
                hidden_size,
                config.num_attention_heads(),
                config.num_key_value_heads(),
                config.head_dim(),
                config.intermediate_size(),
                theta,
                config.rms_norm_eps(),
                None,
                config.has_qk_norm(),
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
            config.num_attention_heads(), // Use num_heads for expanded KV cache
            config.head_dim(),
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
            tp_config: None,
        })
    }

    pub fn new_with_tp(
        config: Qwen3Config,
        tp_config: Option<TensorParallelConfig>,
        num_kv_blocks: usize,
    ) -> CandleResult<Self> {
        let device = tp_config
            .as_ref()
            .map(|tp| tp.local_device())
            .unwrap_or(Device::Cpu);

        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        let theta = config.rope_theta();
        for _ in 0..config.num_hidden_layers() {
            let layer = TransformerBlock::new_with_tp(
                hidden_size,
                config.num_attention_heads(),
                config.num_key_value_heads(), // Use num_kv_heads for KV cache and GQA
                config.head_dim(),
                config.intermediate_size(),
                theta,
                config.rms_norm_eps(),
                tp_config.clone(),
                config.has_qk_norm(),
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
            config.num_attention_heads(), // Use num_heads for expanded KV cache
            config.head_dim(),
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
            tp_config,
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
        num_kv_blocks: usize,
        _kv_quantization: bool,
    ) -> CandleResult<Self> {
        let _vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights.get(embed_key).cloned();
        let embed_tokens = if let Some(w) = embed_weight.as_ref() {
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

            let q_norm_key = format!("model.layers.{}.self_attn.q_norm.weight", i);
            let k_norm_key = format!("model.layers.{}.self_attn.k_norm.weight", i);
            let q_norm_weight = weights.get(&q_norm_key).cloned();
            let k_norm_weight = weights.get(&k_norm_key).cloned();

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
                q_norm_weight,
                k_norm_weight,
            ));

            let theta = config.rope_theta();
            let layer = TransformerBlock::new_with_weights(
                hidden_size,
                config.num_attention_heads(), // num_heads
                config.num_key_value_heads(), // num_kv_heads for GQA
                config.head_dim(),
                config.intermediate_size(),
                theta,
                config.rms_norm_eps(),
                config.has_qk_norm(),
                layer_weights,
            )?;
            layers.push(layer);
        }

        let norm_key = "model.norm.weight";
        let norm = if let Some(w) = weights.get(norm_key) {
            let bias = Tensor::zeros(w.dim(0).unwrap_or(hidden_size), w.dtype(), w.device())?;
            LayerNorm::new(w.clone(), bias, config.rms_norm_eps())
        } else {
            return Err(candle_core::Error::msg(format!(
                "Missing weight: {}",
                norm_key
            )));
        };

        let lm_head = if config.tie_word_embeddings() {
            if let Some(embed_w) = embed_weight.as_ref() {
                Linear::new(embed_w.clone(), None)
            } else {
                return Err(candle_core::Error::msg(
                    "tie_word_embeddings is true but embed_tokens.weight not found".to_string(),
                ));
            }
        } else if let Some(w) = Self::get_weight(
            &weights,
            &["lm_head.weight", "output.weight", "model.lm_head.weight"],
        ) {
            Linear::new(w.clone(), None)
        } else {
            return Err(candle_core::Error::msg(
                "Missing lm_head.weight and tie_word_embeddings is false".to_string(),
            ));
        };

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_attention_heads(), // Use num_heads for expanded KV cache
            config.head_dim(),
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
            tp_config: None,
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
            tracing::warn!(
                num_computed = num_computed_tokens,
                positions = ?positions,
                is_prefill = is_prefill,
                "Empty tokens received"
            );
            let hidden_size = self.config.hidden_size();
            let _dummy = Tensor::zeros((1, 1, hidden_size), candle_core::DType::F32, &self.device)
                .map_err(|e| EngineError::new(e.to_string()))?;
            let logits = Tensor::zeros(
                (1, 1, self.config.vocab_size()),
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
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                hidden = layer
                    .forward_prefill(&hidden, &mut self.kv_cache, layer_idx, block_ids, positions)
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }
        } else {
            let position = positions[0]; // positions contains the current decode position
            let decode_position = [position];
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
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

    pub fn forward_quantized_demo(
        &mut self,
        input_tokens: &[TokenId],
    ) -> EngineResult<(Tensor, usize)> {
        let positions: Vec<usize> = (0..input_tokens.len()).collect();
        self.forward_with_cache(input_tokens, 0, &[0], &positions, true)
            .map_err(|e| EngineError::new(e.to_string()))
    }

    fn stack_tokens(&self, tokens: &[Vec<TokenId>]) -> EngineResult<Tensor> {
        let batch_size = tokens.len();
        let token_ids: Vec<u32> = tokens
            .iter()
            .map(|t| t.last().copied().unwrap_or(0))
            .collect();

        Tensor::from_slice(&token_ids, &[batch_size], &self.device)
            .map_err(|e| EngineError::new(e.to_string()))
    }
}

impl ModelBackend for Qwen3Model {
    fn forward_with_cache(
        &mut self,
        input_tokens: &[TokenId],
        num_computed: usize,
        kv_block_ids: &[usize],
        positions: &[usize],
        is_prefill: bool,
    ) -> EngineResult<(Tensor, usize)> {
        Qwen3Model::forward_with_cache(
            self,
            input_tokens,
            num_computed,
            kv_block_ids,
            positions,
            is_prefill,
        )
    }

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

        // Group indices by prefill/decode status
        let mut prefill_indices: Vec<usize> = vec![];
        let mut decode_indices: Vec<usize> = vec![];

        for (i, &is_pf) in is_prefill.iter().enumerate() {
            if is_pf {
                prefill_indices.push(i);
            } else {
                decode_indices.push(i);
            }
        }

        let mut next_tokens = vec![0u32; seq_ids.len()];

        // Process prefill sequences
        if !prefill_indices.is_empty() {
            for &idx in &prefill_indices {
                let tokens = &input_tokens[idx];
                let pos = &positions[idx];
                let blocks = &kv_block_ids[idx];
                let computed = num_computed_tokens[idx];

                let (logits, _) = self
                    .forward_with_cache(
                        tokens, computed, blocks, pos, true, // is_prefill
                    )
                    .map_err(|e| EngineError::new(e.to_string()))?;

                use candle_core::D;
                // logits shape: [batch=1, seq_len, vocab_size]
                // For prefill, take the last token's logits: narrow -> squeeze -> argmax
                let seq_len = logits.dims()[1];
                let last_logits = logits
                    .narrow(1, seq_len - 1, 1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?;

                // argmax on last dim gives scalar
                let next = last_logits
                    .argmax(D::Minus1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec0::<u32>()
                    .map_err(|e| EngineError::new(e.to_string()))?;
                next_tokens[idx] = next;
            }
        }

        // Process decode sequences
        if !decode_indices.is_empty() {
            for &idx in &decode_indices {
                let tokens = &input_tokens[idx];
                let pos = &positions[idx];
                let blocks = &kv_block_ids[idx];
                let computed = num_computed_tokens[idx];

                let (logits, _) = self
                    .forward_with_cache(
                        tokens, computed, blocks, pos, false, // is_decode
                    )
                    .map_err(|e| EngineError::new(e.to_string()))?;

                use candle_core::D;
                // logits shape: [batch=1, seq=1, vocab_size] for decode (3D)
                // squeeze batch and seq, then argmax
                let next = logits
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(0)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .argmax(D::Minus1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec0::<u32>()
                    .map_err(|e| EngineError::new(e.to_string()))?;
                next_tokens[idx] = next;
            }
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
        if seq_ids.is_empty() {
            return Ok(vec![]);
        }

        let mut results = Vec::with_capacity(seq_ids.len());

        for i in 0..seq_ids.len() {
            let tokens = &input_tokens[i];
            let pos = &positions[i];
            let blocks = &kv_block_ids[i];
            let computed = num_computed_tokens[i];
            let pf = is_prefill[i];

            let (logits, _) = self.forward_with_cache(tokens, computed, blocks, pos, pf)?;

            // logits shape: [batch, seq_len, vocab_size]
            // Take last token's logits for sampling
            let logits_vec = if pf {
                // Prefill: take last position
                let seq_len = logits.dims()[1];
                logits
                    .narrow(1, seq_len - 1, 1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .squeeze(1)
                    .map_err(|e| EngineError::new(e.to_string()))?
                    .to_vec1::<f32>()
                    .map_err(|e| EngineError::new(e.to_string()))?
            } else {
                // Decode: squeeze batch and seq dimensions
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
        let mut embeddings = Vec::with_capacity(input_tokens.len());
        let hidden_size = self.config.hidden_size();

        for tokens in input_tokens {
            if tokens.is_empty() {
                embeddings.push(vec![0.0; hidden_size]);
                continue;
            }

            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let hidden = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let mut hidden = hidden
                .unsqueeze(0)
                .map_err(|e| EngineError::new(e.to_string()))?;

            for layer in self.layers.iter_mut() {
                hidden = layer
                    .forward_prefill(
                        &hidden,
                        &mut self.kv_cache,
                        0,
                        &[],
                        &(0..tokens.len()).collect::<Vec<_>>(),
                    )
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }

            hidden = self
                .norm
                .forward(&hidden)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let hidden = hidden
                .squeeze(0)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let pooled: Vec<f32> = hidden
                .mean(0)
                .map_err(|e| EngineError::new(e.to_string()))?
                .flatten_all()
                .map_err(|e| EngineError::new(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| EngineError::new(e.to_string()))?;

            embeddings.push(pooled);
        }

        Ok(embeddings)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_config::Qwen3Config;
    use vllm_traits::ModelBackend;

    #[test]
    fn test_qwen3_model_forward_cpu() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(128),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        // Test forward with single token
        let kv_block_ids = vec![vec![0usize]];
        let num_computed_tokens = vec![0usize];
        let is_prefill = vec![true];

        let output = model
            .forward(
                &[1],
                &[vec![42]],
                &[vec![0]],
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();
        assert_eq!(output.next_tokens.len(), 1);
        assert!(output.next_tokens[0] < 1000);
    }

    #[test]
    fn test_qwen3_model_forward_qk_norm() {
        // Qwen3-0.6B uses q_norm/k_norm
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(128),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            has_qk_norm: Some(true),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let kv_block_ids = vec![vec![0usize]];
        let num_computed_tokens = vec![0usize];
        let is_prefill = vec![true];

        let output = model
            .forward(
                &[1],
                &[vec![42]],
                &[vec![0]],
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();
        assert_eq!(output.next_tokens.len(), 1);
    }

    #[test]
    fn test_qwen3_model_custom_head_dim() {
        // Qwen3-0.6B: hidden=1024, heads=16, head_dim=128 (not 1024/16=64)
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(1024),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(8),
            intermediate_size: Some(3072),
            head_dim: Some(128),
            has_qk_norm: Some(true),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let kv_block_ids = vec![vec![0usize]];
        let num_computed_tokens = vec![0usize];
        let is_prefill = vec![true];

        let output = model
            .forward(
                &[1],
                &[vec![42]],
                &[vec![0]],
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();
        assert_eq!(output.next_tokens.len(), 1);
    }

    #[test]
    fn test_qwen3_model_batch_forward() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(128),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        // Test with batch size 3
        let seq_ids = vec![1u64, 2, 3];
        let input_tokens = vec![vec![1], vec![2], vec![3]];
        let positions = vec![vec![0], vec![0], vec![0]];
        let kv_block_ids = vec![vec![0usize], vec![0], vec![0]];
        let num_computed_tokens = vec![0usize, 0, 0];
        let is_prefill = vec![true, true, true];

        let output = model
            .forward(
                &seq_ids,
                &input_tokens,
                &positions,
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();

        assert_eq!(output.seq_ids.len(), 3);
        assert_eq!(output.next_tokens.len(), 3);
    }

    #[test]
    fn test_embed_single_text() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(128),
            num_hidden_layers: Some(1),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let input_tokens = vec![vec![1u32, 2, 3, 4, 5]];
        let positions = vec![vec![0, 1, 2, 3, 4]];

        let embeddings = model.embed(&input_tokens, &positions).unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 128);
    }

    #[test]
    fn test_embed_batch() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(128),
            num_hidden_layers: Some(1),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let input_tokens = vec![vec![1u32, 2, 3], vec![4u32, 5, 6, 7, 8], vec![9u32, 10]];
        let positions = vec![vec![0, 1, 2], vec![0, 1, 2, 3, 4], vec![0, 1]];

        let embeddings = model.embed(&input_tokens, &positions).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 128);
        }
    }

    #[test]
    fn test_embed_empty_tokens() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(128),
            num_hidden_layers: Some(1),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let input_tokens = vec![vec![]];
        let positions = vec![vec![]];

        let embeddings = model.embed(&input_tokens, &positions).unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 128);
    }

    #[test]
    fn test_qwen3_decode_mode_with_gqa() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(1024),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(8),
            intermediate_size: Some(3072),
            head_dim: Some(128),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let seq_ids = vec![1u64];
        let input_tokens = vec![vec![42]];
        let positions = vec![vec![7]];
        let kv_block_ids = vec![vec![0usize]];
        let num_computed_tokens = vec![7usize];
        let is_prefill = vec![false];

        let output = model
            .forward(
                &seq_ids,
                &input_tokens,
                &positions,
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();
        assert_eq!(output.next_tokens.len(), 1);
    }

    #[test]
    fn test_qwen3_decode_then_continue() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(256),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(512),
            head_dim: Some(64),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let block_ids = vec![vec![0usize]];
        let num_computed = vec![0usize];
        let is_prefill = vec![false];
        let positions = vec![vec![0]];
        let tokens = vec![vec![42u32]];

        let output = model
            .forward(
                &[1],
                &tokens,
                &positions,
                &block_ids,
                &num_computed,
                &is_prefill,
            )
            .unwrap();

        assert_eq!(output.next_tokens.len(), 1);
    }

    #[test]
    fn test_qwen3_kv_cache_gqa_dimensions() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(512),
            num_hidden_layers: Some(1),
            num_attention_heads: Some(8),
            num_key_value_heads: Some(2),
            intermediate_size: Some(1024),
            head_dim: Some(64),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 8).unwrap();

        let seq_ids = vec![1u64, 2u64];
        let input_tokens = vec![vec![1u32, 2, 3], vec![4u32, 5]];
        let positions = vec![vec![0, 1, 2], vec![0, 1]];
        let kv_block_ids = vec![vec![0, 1], vec![2, 3]];
        let num_computed_tokens = vec![0usize, 0];
        let is_prefill = vec![true, true];

        let output = model
            .forward(
                &seq_ids,
                &input_tokens,
                &positions,
                &kv_block_ids,
                &num_computed_tokens,
                &is_prefill,
            )
            .unwrap();

        assert_eq!(output.seq_ids.len(), 2);
        assert_eq!(output.next_tokens.len(), 2);
    }

    #[test]
    fn test_qwen3_decode_logits_2d_shape() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(256),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(512),
            head_dim: Some(64),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let _seq_ids = [1u64];
        let _input_tokens = [vec![42u32]];
        let _positions = [vec![7]];
        let _kv_block_ids = [vec![0usize]];
        let _num_computed_tokens = [7usize];
        let _is_prefill = [false];

        let (logits, _hidden_token) = model
            .forward_with_cache(&[42], 7, &[0], &[7], false)
            .unwrap();

        assert_eq!(
            logits.dims(),
            &[1, 1, 1000],
            "Decode logits should be [1, 1, vocab_size]"
        );
    }

    #[test]
    fn test_qwen3_prefill_logits_3d_shape() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(256),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(512),
            head_dim: Some(64),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 1024).unwrap();

        let (logits, _hidden_token) = model
            .forward_with_cache(
                &[1, 2, 3, 4, 5],
                0,
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3, 4],
                true,
            )
            .unwrap();

        assert_eq!(
            logits.dims(),
            &[1, 5, 1000],
            "Prefill logits should be [1, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_qwen3_decode_sequential_generation() {
        let config = Qwen3Config {
            vocab_size: Some(500),
            hidden_size: Some(256),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(512),
            head_dim: Some(64),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 32).unwrap();

        let block_size = 8;

        for step in 0..20 {
            let block_id = step / block_size;
            let num_computed = step;
            let position = step;

            let (logits, _hidden_token) = model
                .forward_with_cache(&[42], num_computed, &[block_id], &[position], false)
                .unwrap();

            assert_eq!(
                logits.dims(),
                &[1, 1, 500],
                "Step {}: logits shape mismatch",
                step
            );

            let next_token: u32 = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .argmax(candle_core::D::Minus1)
                .unwrap()
                .to_vec0::<u32>()
                .unwrap();
            assert!(
                next_token < 500,
                "Step {}: next token {} out of range",
                step,
                next_token
            );
        }
    }

    #[test]
    fn test_qwen3_decode_with_gqa_expanded_kv_cache() {
        let config = Qwen3Config {
            vocab_size: Some(1000),
            hidden_size: Some(1024),
            num_hidden_layers: Some(3),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(8),
            intermediate_size: Some(3072),
            head_dim: Some(128),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 16).unwrap();

        let output = model
            .forward(
                &[1],
                &[vec![42u32]],
                &[vec![5]],
                &[vec![0usize]],
                &[5],
                &[false],
            )
            .unwrap();

        assert_eq!(output.next_tokens.len(), 1);
        assert!(output.next_tokens[0] < 1000);
    }

    #[test]
    fn test_qwen3_mixed_prefill_and_decode() {
        let config = Qwen3Config {
            vocab_size: Some(500),
            hidden_size: Some(128),
            num_hidden_layers: Some(2),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            head_dim: Some(32),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 16).unwrap();

        let _ = model
            .forward(
                &[1],
                &[vec![1u32, 2, 3, 4, 5]],
                &[vec![0, 1, 2, 3, 4]],
                &[vec![0, 1, 2, 3, 4]],
                &[0, 1, 2, 3, 4],
                &[true],
            )
            .unwrap();

        for step in 5..10 {
            let block_id = step / 8;
            let output = model
                .forward(
                    &[1],
                    &[vec![42u32]],
                    &[vec![step]],
                    &[vec![block_id]],
                    &[step],
                    &[false],
                )
                .unwrap();

            assert_eq!(output.next_tokens.len(), 1);
            assert!(output.next_tokens[0] < 500);
        }
    }

    #[test]
    fn test_forward_logits_decode_mode() {
        let config = Qwen3Config {
            vocab_size: Some(300),
            hidden_size: Some(128),
            num_hidden_layers: Some(1),
            num_attention_heads: Some(4),
            num_key_value_heads: Some(2),
            intermediate_size: Some(256),
            head_dim: Some(32),
            ..Default::default()
        };

        let device = Device::Cpu;
        let mut model = Qwen3Model::new(config, device, 8).unwrap();

        let result = model
            .forward_logits(
                &[1],
                &[vec![42u32]],
                &[vec![5]],
                &[vec![0usize]],
                &[5],
                &[false],
            )
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 300);
    }
}
