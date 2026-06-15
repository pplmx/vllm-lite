//! Shared checkpoint weight resolution for causal and hybrid LMs.

use std::collections::HashMap;

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::Linear;

/// Resolve the final decoder norm weight from common HF key layouts.
pub(crate) fn load_final_norm_weight(weights: &HashMap<String, Tensor>) -> Option<Tensor> {
    const KEYS: &[&str] = &[
        "model.norm.weight",
        "model.language_model.norm.weight",
        "model.final_layernorm.weight",
    ];
    KEYS.iter().find_map(|key| weights.get(*key).cloned())
}

/// Load `lm_head` with tied-embedding and alternate HF key fallbacks.
pub(crate) fn load_lm_head(
    weights: &HashMap<String, Tensor>,
    embed_weight: Tensor,
    tie_word_embeddings: bool,
) -> CandleResult<Linear> {
    if tie_word_embeddings {
        Ok(Linear::new(embed_weight, None))
    } else {
        let lm_weight = weights
            .get("lm_head.weight")
            .cloned()
            .or_else(|| weights.get("output.weight").cloned())
            .or_else(|| weights.get("model.lm_head.weight").cloned())
            .or_else(|| weights.get("model.embed_tokens.weight").cloned())
            .ok_or_else(|| candle_core::Error::msg("Missing lm_head.weight"))?;
        Ok(Linear::new(lm_weight, None))
    }
}
