//! Unified decoder layer loop with optional per-layer auxiliary state (e.g. GDN).

use crate::components::decoder_block::{PagedDecoderBlock, RopeGqaDecoderBlock};
use crate::components::gated_delta::GatedDeltaState;
use crate::paged_tensor::PagedKvCache;
use candle_core::{Result, Tensor};
use vllm_traits::Result as EngineResult;

use super::map_candle;

#[derive(Debug)]
/// Per-layer auxiliary state passed through the layer loop.
pub enum LayerAuxMut<'a> {
    /// Gated-delta-net recurrent state for Qwen3.5 linear-attention layers.
    Gdn(&'a mut [Option<GatedDeltaState>]),
}
#[derive(Debug)]

/// Context shared across all layers in a single forward pass.
pub struct LayerCtx<'a> {
    pub kv_cache: &'a mut PagedKvCache,
    pub block_ids: &'a [usize],
    pub positions: &'a [usize],
    pub num_computed_tokens: usize,
    pub is_prefill: bool,
    pub aux: Option<LayerAuxMut<'a>>,
}

/// Production decoder layer: paged-KV prefill/decode with optional aux state.
pub trait DecoderLayer {
    /// Run the prefill path: process the full prompt and cache its KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_prefill(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> Result<Tensor>;

    /// Run the decode path: process one new token against cached KV.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    fn forward_decode(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> Result<Tensor>;
}

impl<L: PagedDecoderBlock> DecoderLayer for L {
    fn forward_prefill(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        self.forward_prefill(x, ctx.kv_cache, layer_idx, ctx.block_ids, ctx.positions)
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let decode_position = [ctx.positions[0]];
        self.forward_decode(
            x,
            ctx.kv_cache,
            layer_idx,
            ctx.block_ids,
            ctx.num_computed_tokens,
            &decode_position,
        )
    }
}

impl DecoderLayer for RopeGqaDecoderBlock {
    fn forward_prefill(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        if ctx.num_computed_tokens > 0 {
            Self::forward_prefill_continue(
                self,
                x,
                ctx.kv_cache,
                layer_idx,
                ctx.block_ids,
                ctx.positions,
                ctx.num_computed_tokens,
            )
        } else {
            Self::forward_prefill(
                self,
                x,
                ctx.kv_cache,
                layer_idx,
                ctx.block_ids,
                ctx.positions,
            )
        }
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let decode_position = [ctx.positions[0]];
        Self::forward_decode(
            self,
            x,
            ctx.kv_cache,
            layer_idx,
            ctx.block_ids,
            ctx.num_computed_tokens,
            &decode_position,
        )
    }
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Run all decoder layers (prefill or single-token decode).
pub fn run_layers<L: DecoderLayer>(
    layers: &[L],
    hidden: Tensor,
    ctx: &mut LayerCtx<'_>,
) -> EngineResult<Tensor> {
    run_layers_upto(layers, hidden, ctx, layers.len())
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Run decoder layers up to (but not including) `upto_layer`.
pub fn run_layers_upto<L: DecoderLayer>(
    layers: &[L],
    mut hidden: Tensor,
    ctx: &mut LayerCtx<'_>,
    upto_layer: usize,
) -> EngineResult<Tensor> {
    let upto = upto_layer.min(layers.len());
    for (layer_idx, layer) in layers.iter().enumerate().take(upto) {
        hidden = map_candle(if ctx.is_prefill {
            layer.forward_prefill(&hidden, ctx, layer_idx)
        } else {
            layer.forward_decode(&hidden, ctx, layer_idx)
        })?;
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::decoder_block::new_block;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device};

    #[test]
    fn test_run_layers_matches_paged_decoder_block() {
        let config = ModelConfig::test_tiny();
        let device = Device::Cpu;
        let layer = new_block(&config, 0).unwrap();
        let layers = vec![layer];
        let mut kv_cache = PagedKvCache::new(
            1,
            config.num_heads,
            config.head_dim,
            32,
            device.clone(),
            false,
        )
        .unwrap();

        let seq_len = 3usize;
        let hidden = Tensor::ones((1, seq_len, config.hidden_size), DType::F32, &device).unwrap();
        let block_ids: Vec<usize> = (0..seq_len).map(|i| i / 32).collect();
        let positions: Vec<usize> = (0..seq_len).collect();

        let mut ctx = LayerCtx {
            kv_cache: &mut kv_cache,
            block_ids: &block_ids,
            positions: &positions,
            num_computed_tokens: 0,
            is_prefill: true,
            aux: None,
        };
        let out = run_layers(&layers, hidden, &mut ctx).unwrap();
        assert_eq!(out.dims(), &[1, seq_len, config.hidden_size]);
    }
}
