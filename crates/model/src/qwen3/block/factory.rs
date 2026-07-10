//! Free factory functions used by the qwen3 model + tests: `new_block`
//! (zero-init constructor for tests) and `block_from_weights`
//! (HuggingFace weight-map loader, used by [`super::TransformerBlock::from_weights`]).

use std::collections::HashMap;

use super::TransformerBlock;
use candle_core::{Result, Tensor};

/// Run the operation (see signature for params and return type).
/// # Errors
///
/// Returns `Err` if the operation fails.
pub fn new_block(config: &crate::config::ModelConfig, _layer_idx: usize) -> Result<TransformerBlock> {
    TransformerBlock::new(
        config.hidden_size,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.rope_theta,
        config.rms_norm_eps,
        None,
        config.has_qk_norm,
    )
}

pub(crate) fn block_from_weights(
    config: &crate::config::ModelConfig,
    layer_idx: usize,
    weights: &HashMap<String, Tensor>,
) -> Result<TransformerBlock> {
    TransformerBlock::from_weights(config, layer_idx, weights)
}
