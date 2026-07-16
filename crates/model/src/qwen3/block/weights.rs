//! `TransformerBlock::from_weights` — build a layer from a HuggingFace
//! weight map. Looks up each component's tensor by convention
//! (`model.layers.{idx}.self_attn.q_proj.weight`, etc.) and delegates
//! to [`super::TransformerBlock::new_with_weights`].

use std::collections::HashMap;

use super::TransformerBlock;
use candle_core::{Result, Tensor};

impl TransformerBlock {
    /// Build from weights.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_weights(
        config: &crate::config::ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let intermediate_size = config.intermediate_size;
        let theta = config.rope_theta;
        let rms_norm_eps = config.rms_norm_eps;
        let has_qk_norm = config.has_qk_norm;

        let get_weight = |keys: &[&str]| -> Option<&Tensor> {
            for key in keys {
                if let Some(w) = weights.get(*key) {
                    return Some(w);
                }
            }
            None
        };

        let q_key = get_weight(&[
            &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            &format!("model.layers.{layer_idx}.attn.q_proj.weight"),
        ]);
        let k_key = get_weight(&[
            &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
            &format!("model.layers.{layer_idx}.attn.k_proj.weight"),
        ]);
        let v_key = get_weight(&[
            &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
            &format!("model.layers.{layer_idx}.attn.v_proj.weight"),
        ]);
        let o_key = get_weight(&[
            &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            &format!("model.layers.{layer_idx}.attn.o_proj.weight"),
        ]);

        let q_norm_key = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
        let k_norm_key = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
        let q_norm_weight = weights.get(&q_norm_key).cloned();
        let k_norm_weight = weights.get(&k_norm_key).cloned();

        let layer_weights = Some((
            q_key.cloned(),
            k_key.cloned(),
            v_key.cloned(),
            o_key.cloned(),
            weights
                .get(&format!("model.layers.{layer_idx}.mlp.gate_proj.weight"))
                .cloned(),
            weights
                .get(&format!("model.layers.{layer_idx}.mlp.up_proj.weight"))
                .cloned(),
            weights
                .get(&format!("model.layers.{layer_idx}.mlp.down_proj.weight"))
                .cloned(),
            weights
                .get(&format!("model.layers.{layer_idx}.input_layernorm.weight"))
                .cloned(),
            weights
                .get(&format!(
                    "model.layers.{layer_idx}.post_attention_layernorm.weight"
                ))
                .cloned(),
            q_norm_weight,
            k_norm_weight,
        ));

        Self::new_with_weights_rope_scaling(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            theta,
            rms_norm_eps,
            config.max_position_embeddings,
            config.rope_scaling.as_ref(),
            has_qk_norm,
            layer_weights,
        )
    }
}
