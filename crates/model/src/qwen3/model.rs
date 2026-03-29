use crate::config::Qwen3Config;
use crate::kv_cache::PagedKvCache;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear};
use vllm_core::engine::ModelBackend;
use vllm_core::error::Result as EngineResult;
use vllm_core::types::{BatchOutput, SeqId, TokenId};

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
    pub fn new(config: Qwen3Config, device: Device) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        let embeddings = Tensor::zeros((vocab_size, hidden_size), DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            let layer = TransformerBlock::new(
                hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                hidden_size / config.num_attention_heads,
                config.intermediate_size,
                config.rms_norm_eps as f64,
                candle_nn::VarBuilder::zeros(DType::F32, &device),
            )?;
            layers.push(layer);
        }

        let norm = candle_nn::layer_norm(
            hidden_size,
            config.rms_norm_eps as f64,
            candle_nn::VarBuilder::zeros(DType::F32, &device),
        )?;
        let lm_head = candle_nn::linear(
            hidden_size,
            vocab_size,
            candle_nn::VarBuilder::zeros(DType::F32, &device),
        )?;

        let kv_cache = PagedKvCache::new(
            config.num_hidden_layers,
            config.num_key_value_heads,
            hidden_size / config.num_attention_heads,
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
}

impl ModelBackend for Qwen3Model {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<BatchOutput> {
        use rand::RngExt;
        let mut rng = rand::rng();
        let next_tokens: Vec<TokenId> = seq_ids
            .iter()
            .map(|_| rng.random_range(0..self.config.vocab_size) as TokenId)
            .collect();

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }
}
