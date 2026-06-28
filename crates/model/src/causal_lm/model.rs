use std::collections::HashMap;

use super::{
    LayerCtx, embed_sequence, embed_with_paged_layers, forward_batch, forward_with_paged_kv,
    greedy_sample_token, logits_to_vector, map_candle, mean_pool_embeddings, run_layers_upto,
};
use crate::components::decoder_block::PagedDecoderBlock;
use crate::components::{LnLayerNorm, RmsNorm};
use crate::config::ModelConfig;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use vllm_traits::{BatchOutput, BlockId, ModelBackend, Result, SeqId, TokenId};

#[derive(Debug)]
/// Generic decoder-only causal language model shell.
pub struct CausalLm<B, Norm, Head> {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<B>,
    norm: Norm,
    lm_head: Head,
    kv_cache: PagedKvCache,
    device: Device,
    embed_through_layers: bool,
}

impl<B, Norm, Head> CausalLm<B, Norm, Head>
where
    B: PagedDecoderBlock,
    Norm: Module,
    Head: Module,
{
    pub const fn with_embed_through_layers(mut self, enabled: bool) -> Self {
        self.embed_through_layers = enabled;
        self
    }

    pub fn forward_with_cache(
        &mut self,
        tokens: &[TokenId],
        num_computed_tokens: usize,
        block_ids: &[BlockId],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        forward_with_paged_kv(
            &self.embed_tokens,
            &self.layers,
            &self.norm,
            &self.lm_head,
            &self.device,
            self.config.vocab_size,
            tokens,
            num_computed_tokens,
            block_ids,
            positions,
            is_prefill,
            &mut self.kv_cache,
        )
    }
}

impl<B> CausalLm<B, LnLayerNorm, Linear>
where
    B: PagedDecoderBlock + Send + Sync,
{
    pub fn new_with_block_fn<F>(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
        mut block_fn: F,
    ) -> CandleResult<Self>
    where
        F: FnMut(&ModelConfig, usize) -> CandleResult<B>,
    {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_fn(&config, layer_idx)?);
        }

        let norm_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let norm_bias = Tensor::zeros(hidden_size, candle_core::DType::F32, &device)?;
        let norm = LnLayerNorm::new(norm_weight, norm_bias, config.rms_norm_eps);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
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
            embed_through_layers: false,
        })
    }

    pub fn from_hf_weights_ln<F>(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
        block_from_weights: F,
    ) -> CandleResult<Self>
    where
        F: Fn(&ModelConfig, usize, &HashMap<String, Tensor>) -> CandleResult<B>,
    {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {embed_key}")))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_from_weights(&config, layer_idx, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {norm_key}")))?;
        let norm_bias = Tensor::zeros(
            norm_weight.dim(0).unwrap_or(hidden_size),
            norm_weight.dtype(),
            norm_weight.device(),
        )?;
        let norm = LnLayerNorm::new(norm_weight, norm_bias, config.rms_norm_eps);

        let lm_head =
            super::weights::load_lm_head(&weights, embed_weight, config.tie_word_embeddings)?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
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
            embed_through_layers: false,
        })
    }
}

impl<B> CausalLm<B, RmsNorm, Linear>
where
    B: PagedDecoderBlock + Send + Sync,
{
    pub fn new_rms<F>(
        config: ModelConfig,
        device: Device,
        num_kv_blocks: usize,
        kv_quantization: bool,
        mut block_fn: F,
    ) -> CandleResult<Self>
    where
        F: FnMut(&ModelConfig, usize) -> CandleResult<B>,
    {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embeddings =
            Tensor::zeros((vocab_size, hidden_size), candle_core::DType::F32, &device)?;
        let embed_tokens = Embedding::new(embeddings, hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_fn(&config, layer_idx)?);
        }

        let norm_weight = Tensor::ones(hidden_size, candle_core::DType::F32, &device)?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
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
            embed_through_layers: false,
        })
    }

    pub fn from_hf_weights_rms<F>(
        config: ModelConfig,
        device: Device,
        weights: HashMap<String, Tensor>,
        num_kv_blocks: usize,
        kv_quantization: bool,
        block_from_weights: F,
    ) -> CandleResult<Self>
    where
        F: Fn(&ModelConfig, usize, &HashMap<String, Tensor>) -> CandleResult<B>,
    {
        let hidden_size = config.hidden_size;
        let num_layers = config.num_layers;

        let embed_key = "model.embed_tokens.weight";
        let embed_weight = weights
            .get(embed_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {embed_key}")))?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_size);

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(block_from_weights(&config, layer_idx, &weights)?);
        }

        let norm_key = "model.norm.weight";
        let norm_weight = weights
            .get(norm_key)
            .cloned()
            .ok_or_else(|| candle_core::Error::msg(format!("Missing {norm_key}")))?;
        let norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        let lm_head =
            super::weights::load_lm_head(&weights, embed_weight, config.tie_word_embeddings)?;

        let kv_cache = PagedKvCache::new(
            num_layers,
            config.num_heads,
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
            embed_through_layers: false,
        })
    }
}

impl<B, Norm, Head> ModelBackend for CausalLm<B, Norm, Head>
where
    B: PagedDecoderBlock + Send + Sync,
    Norm: Module + Send + Sync,
    Head: Module + Send + Sync,
{
    fn forward_with_cache(
        &mut self,
        input_tokens: &[TokenId],
        num_computed: usize,
        kv_block_ids: &[usize],
        positions: &[usize],
        is_prefill: bool,
    ) -> Result<(Tensor, usize)> {
        self.forward_with_cache(
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
    ) -> Result<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let (logits, _) = self.forward_with_cache(
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                prefill,
            )?;
            greedy_sample_token(&logits, prefill)
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
    ) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(seq_ids.len());
        for i in 0..seq_ids.len() {
            let (logits, _) = self.forward_with_cache(
                &input_tokens[i],
                num_computed_tokens[i],
                &kv_block_ids[i],
                &positions[i],
                is_prefill[i],
            )?;
            results.push(logits_to_vector(&logits, is_prefill[i])?);
        }
        Ok(results)
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        if self.embed_through_layers {
            embed_with_paged_layers(
                &self.embed_tokens,
                &self.layers,
                &self.norm,
                &self.device,
                self.config.hidden_size,
                &mut self.kv_cache,
                input_tokens,
                positions,
            )
        } else {
            input_tokens
                .iter()
                .map(|tokens| {
                    mean_pool_embeddings(
                        &self.embed_tokens,
                        tokens,
                        &self.device,
                        self.config.hidden_size,
                    )
                })
                .collect()
        }
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

    fn forward_to_layer(
        &mut self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
        kv_block_ids: &[Vec<usize>],
        num_computed_tokens: &[usize],
        is_prefill: &[bool],
        upto_layer: usize,
    ) -> Result<BatchOutput> {
        forward_batch(seq_ids, is_prefill, |i, prefill| {
            let tokens = &input_tokens[i];
            if tokens.is_empty() {
                return Ok(0);
            }

            let hidden = embed_sequence(&self.embed_tokens, tokens, &self.device, prefill)?;
            let mut ctx = LayerCtx {
                kv_cache: &mut self.kv_cache,
                block_ids: &kv_block_ids[i],
                positions: &positions[i],
                num_computed_tokens: num_computed_tokens[i],
                is_prefill: prefill,
                aux: None,
            };
            let hidden = run_layers_upto(&self.layers, hidden, &mut ctx, upto_layer)?;
            let hidden = map_candle(self.norm.forward(&hidden))?;
            let logits = map_candle(self.lm_head.forward(&hidden))?;
            greedy_sample_token(&logits, prefill)
        })
    }
}
