//! Hybrid decoder blocks (GDN linear + full attention).

#![allow(clippy::too_many_arguments)]

mod full;
mod linear;

pub use full::FullAttentionBlock35;
pub use linear::LinearAttentionBlock;

use crate::causal_lm::{DecoderLayer, LayerAuxMut, LayerCtx};
use candle_core::{Result as CandleResult, Tensor};

/// `HybridBlock`: hybrid block enumeration.
pub enum HybridBlock {
    Linear(LinearAttentionBlock),
    Full(FullAttentionBlock35),
}

impl DecoderLayer for HybridBlock {
    fn forward_prefill(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> CandleResult<Tensor> {
        let LayerCtx {
            kv_cache,
            block_ids,
            positions,
            aux,
            ..
        } = ctx;
        let gdn_state = match aux {
            Some(LayerAuxMut::Gdn(states)) => &mut states[layer_idx],
            None => {
                return Err(candle_core::Error::msg(format!(
                    "missing GDN aux state for hybrid layer {layer_idx}"
                )));
            }
        };
        match self {
            Self::Linear(b) => {
                let (out, state) = b.forward_prefill(x)?;
                *gdn_state = Some(state);
                Ok(out)
            }
            Self::Full(b) => b.forward_prefill(x, kv_cache, layer_idx, block_ids, positions),
        }
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        ctx: &mut LayerCtx<'_>,
        layer_idx: usize,
    ) -> CandleResult<Tensor> {
        let LayerCtx {
            kv_cache,
            block_ids,
            positions,
            num_computed_tokens,
            aux,
            ..
        } = ctx;
        let gdn_state = match aux {
            Some(LayerAuxMut::Gdn(states)) => &mut states[layer_idx],
            None => {
                return Err(candle_core::Error::msg(format!(
                    "missing GDN aux state for hybrid layer {layer_idx}"
                )));
            }
        };
        let decode_pos = [positions[0]];
        match self {
            Self::Linear(b) => {
                let state = gdn_state.as_mut().ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "missing GDN state for linear layer {layer_idx}"
                    ))
                })?;
                b.forward_decode(x, state)
            }
            Self::Full(b) => b.forward_decode(
                x,
                kv_cache,
                layer_idx,
                block_ids,
                *num_computed_tokens,
                &decode_pos,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn test_hybrid_block_enum_creation() {
        let device = Device::Cpu;
        let hidden_size = 128;

        let gdn = crate::qwen3_5::config::GdnLinearConfig::legacy_heuristic(hidden_size);
        let linear =
            LinearAttentionBlock::new(hidden_size, gdn, VarBuilder::zeros(DType::F32, &device))
                .unwrap();

        let linear_block = HybridBlock::Linear(linear);
        match linear_block {
            HybridBlock::Linear(_) => {}
            _ => panic!("Expected Linear variant"),
        }
    }
}
