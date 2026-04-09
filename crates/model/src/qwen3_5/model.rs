#![allow(clippy::all, unused)]
use crate::kv_cache::PagedKvCache;
use crate::qwen3_5::ssm::{MambaBlock, SSMConfig};
use crate::qwen3_config::Qwen3Config;
use candle_core::{Device, Module, Result as CandleResult, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use std::collections::HashMap;
use vllm_traits::{BatchOutput, SeqId, TokenId};
use vllm_traits::{ModelBackend, Result as EngineResult};

pub type EngineError = vllm_traits::ModelError;

pub struct Qwen35Model {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<MambaBlock>,
    norm: LayerNorm,
    lm_head: Linear,
    kv_cache: PagedKvCache,
    device: Device,
}

impl Qwen35Model {
    pub fn new(config: Qwen3Config, device: Device, num_kv_blocks: usize) -> CandleResult<Self> {
        let vocab_size = config.vocab_size();
        let hidden_size = config.hidden_size();

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let num_layers = config.num_hidden_layers();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let ssm_config = SSMConfig::new(hidden_size);
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(MambaBlock::new(
                hidden_size,
                ssm_config.d_state,
                vb.clone(),
            )?);
        }

        let norm = candle_nn::layer_norm(hidden_size, config.rms_norm_eps(), vb.clone())?;
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb)?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers(),
            config.num_key_value_heads(),
            hidden_size / config.num_attention_heads(),
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
        config: Qwen3Config,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
    ) -> CandleResult<Self> {
        let mut model = Self::new(config.clone(), device.clone(), num_kv_blocks)?;

        // Load embed_tokens
        if let Some(w) = weights.get("model.language_model.embed_tokens.weight") {
            model.embed_tokens = Embedding::new(w.clone(), w.dims()[1]);
            println!("Loaded embed_tokens");
        }

        // TODO: Load layer weights when from_weights is implemented for MambaBlock

        Ok(model)
    }
}

impl ModelBackend for Qwen35Model {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
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
                .map_err(|e| EngineError::new(e.to_string()))?;

            let hidden = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let mut hidden = hidden;
            for layer in &mut self.layers {
                hidden = layer
                    .forward(&hidden)
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }

            hidden = self
                .norm
                .forward(&hidden)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let logits = self
                .lm_head
                .forward(&hidden)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let seq_len = logits.dims()[0];
            let last_logits = logits
                .get(seq_len - 1)
                .map_err(|e| EngineError::new(e.to_string()))?;

            let max_idx = last_logits
                .argmax(0)
                .map_err(|e| EngineError::new(e.to_string()))?
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
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<Vec<Vec<f32>>> {
        let vocab_size = self.config.vocab_size();
        Ok(input_tokens
            .iter()
            .map(|t| vec![0.0; vocab_size * t.len()])
            .collect())
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

            let mut hidden = self
                .embed_tokens
                .forward(&token_tensor)
                .map_err(|e| EngineError::new(e.to_string()))?;

            for layer in &mut self.layers {
                hidden = layer
                    .forward(&hidden)
                    .map_err(|e| EngineError::new(e.to_string()))?;
            }

            hidden = self
                .norm
                .forward(&hidden)
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
}
