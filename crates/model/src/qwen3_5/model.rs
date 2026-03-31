#![allow(clippy::all, unused)]
use crate::config::Qwen3Config;
use crate::kv_cache::PagedKvCache;
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use std::collections::HashMap;
use vllm_core::engine::ModelBackend;
use vllm_core::error::{EngineError, Result as EngineResult};
use vllm_core::types::{BatchOutput, SeqId, TokenId};

pub struct Qwen35Model {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<MambaBlock>,
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

pub struct MambaBlock {
    linear: Linear,
}

impl MambaBlock {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> CandleResult<Self> {
        let linear = candle_nn::linear(hidden_size, hidden_size, vb)?;
        Ok(Self { linear })
    }
}

impl Qwen35Model {
    pub fn new(config: Qwen3Config, device: Device) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let num_layers = config.num_hidden_layers();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(MambaBlock::new(hidden_size, vb.clone())?);
        }

        let norm = candle_nn::layer_norm(hidden_size, config.rms_norm_eps(), vb.clone())?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb)?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            hidden_size / config.num_attention_heads(),
            1024,
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
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
    ) -> CandleResult<Self> {
        let mut model = Self::new(config.clone(), device.clone())?;

        let embed_key = "model.language_model.embed_tokens.weight";
        if let Some(w) = weights.get(embed_key) {
            model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
            println!("Loaded embed_tokens from {}", embed_key);
        }

        eprintln!("Warning: Qwen3.5 Mamba implementation is simplified (placeholder)");
        Ok(model)
    }
}

impl ModelBackend for Qwen35Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<BatchOutput> {
        if seq_ids.is_empty() {
            return Ok(BatchOutput {
                seq_ids: vec![],
                next_tokens: vec![],
            });
        }

        let batch_size = seq_ids.len();
        let vocab_size = self.config.vocab_size();

        let mut next_tokens = Vec::with_capacity(batch_size);

        for (i, tokens) in input_tokens.iter().enumerate() {
            if tokens.is_empty() {
                next_tokens.push(0);
                continue;
            }

            let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let hidden = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let mut hidden = hidden;
            for layer in &self.layers {
                hidden = layer
                    .linear
                    .forward(&hidden)
                    .map_err(|e| EngineError::ModelError(e.to_string()))?;
            }

            hidden = self
                .norm
                .forward(&hidden)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let logits = self
                .lm_head
                .forward(&hidden)
                .map_err(|e| EngineError::ModelError(e.to_string()))?;

            let seq_len = logits.dims()[0];
            let last_logits = logits
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

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<Vec<Vec<f32>>> {
        let vocab_size = self.config.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|t| vec![0.0; vocab_size * t.len()])
            .collect())
    }
}
