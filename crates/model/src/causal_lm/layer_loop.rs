//! Unified decoder layer loop with optional per-layer auxiliary state (e.g. GDN).

use crate::components::decoder_block::PagedDecoderBlock;
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
        if ctx.num_computed_tokens > 0 {
            self.forward_prefill_continue(
                x,
                ctx.kv_cache,
                layer_idx,
                ctx.block_ids,
                ctx.positions,
                ctx.num_computed_tokens,
            )
        } else {
            self.forward_prefill(x, ctx.kv_cache, layer_idx, ctx.block_ids, ctx.positions)
        }
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
    use vllm_traits::BLOCK_SIZE;

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
        let block_ids: Vec<usize> = (0..seq_len.div_ceil(BLOCK_SIZE)).collect();
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

    #[test]
    fn test_run_layers_transformer_block_chunked_prefill() {
        use crate::qwen3::block::factory::new_block;

        let config = ModelConfig::test_tiny();
        let device = Device::Cpu;
        let layer = new_block(&config, 0).unwrap();
        let layers = vec![layer];

        let seq_len = 6usize;
        let hidden =
            Tensor::ones((1, seq_len, config.hidden_size), DType::F32, &device).unwrap();
        let block_ids: Vec<usize> = (0..seq_len.div_ceil(BLOCK_SIZE)).collect();
        let positions: Vec<usize> = (0..seq_len).collect();

        let mut full_cache = PagedKvCache::new(
            1,
            config.num_heads,
            config.head_dim,
            32,
            device.clone(),
            false,
        )
        .unwrap();
        let mut full_ctx = LayerCtx {
            kv_cache: &mut full_cache,
            block_ids: &block_ids,
            positions: &positions,
            num_computed_tokens: 0,
            is_prefill: true,
            aux: None,
        };
        let full_out = run_layers(&layers, hidden.clone(), &mut full_ctx).unwrap();

        let mut chunked_cache = PagedKvCache::new(
            1,
            config.num_heads,
            config.head_dim,
            32,
            device.clone(),
            false,
        )
        .unwrap();
        let chunk1 = hidden.narrow(1, 0, 3).unwrap();
        let pos1: Vec<usize> = (0..3).collect();
        let mut ctx1 = LayerCtx {
            kv_cache: &mut chunked_cache,
            block_ids: &block_ids,
            positions: &pos1,
            num_computed_tokens: 0,
            is_prefill: true,
            aux: None,
        };
        run_layers(&layers, chunk1, &mut ctx1).unwrap();

        let chunk2 = hidden.narrow(1, 3, 3).unwrap();
        let pos2: Vec<usize> = (3..6).collect();
        let mut ctx2 = LayerCtx {
            kv_cache: &mut chunked_cache,
            block_ids: &block_ids,
            positions: &pos2,
            num_computed_tokens: 3,
            is_prefill: true,
            aux: None,
        };
        let cont_out = run_layers(&layers, chunk2, &mut ctx2).unwrap();

        let full_last = full_out.narrow(1, seq_len - 1, 1).unwrap();
        let cont_last = cont_out.narrow(1, 2, 1).unwrap();
        let diff = (full_last - cont_last)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_vec0::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "TransformerBlock chunked prefill diverged: diff={diff}"
        );
    }
}
