//! Mixtral block (Transformer layer with MoE).

#![allow(dead_code)]

use std::collections::HashMap;

use crate::config::ModelConfig;
use crate::mixtral::sparse_moe::MixtralSparseMoe;
use crate::qwen3::attention::GqaAttention;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder};

pub struct MixtralBlock {
    attention: GqaAttention,
    mlp: MixtralSparseMoe,
    input_layernorm: Linear,
    post_attention_layernorm: Linear,
    sliding_window: usize,
}

impl MixtralBlock {
    pub fn new(config: &ModelConfig, _layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let rope_theta = config.rope_theta;
        let _rms_norm_eps = config.rms_norm_eps;
        let sliding_window = config.sliding_window.unwrap_or(4096);

        let num_experts = config.num_experts.unwrap_or(8);
        let expert_intermediate_size = config
            .expert_intermediate_size
            .unwrap_or(config.intermediate_size);
        let top_k = config.top_k_experts.unwrap_or(2);

        let vb = VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu);

        let input_layernorm =
            candle_nn::linear(hidden_size, hidden_size, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::linear(hidden_size, hidden_size, vb.pp("post_attention_layernorm"))?;

        let attention = GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            Some(vb.clone()),
            crate::components::AttentionConfig::default(),
            false,
        )?;

        let mlp = MixtralSparseMoe::new(
            hidden_size,
            num_experts,
            expert_intermediate_size,
            top_k,
            vb,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            sliding_window,
        })
    }

    pub fn from_weights(
        config: &ModelConfig,
        layer_idx: usize,
        weights: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let rope_theta = config.rope_theta;
        let _rms_norm_eps = config.rms_norm_eps;
        let sliding_window = config.sliding_window.unwrap_or(4096);

        let num_experts = config.num_experts.unwrap_or(8);
        let expert_intermediate_size = config
            .expert_intermediate_size
            .unwrap_or(config.intermediate_size);
        let top_k = config.top_k_experts.unwrap_or(2);

        let q_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .cloned();
        let k_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .cloned();
        let v_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .cloned();
        let o_w = weights
            .get(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                layer_idx
            ))
            .cloned();
        let input_ln_w = weights
            .get(&format!(
                "model.layers.{}.input_layernorm.weight",
                layer_idx
            ))
            .cloned();
        let post_attn_ln_w = weights
            .get(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                layer_idx
            ))
            .cloned();

        let q_w = match q_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing q_proj weight")),
        };
        let k_w = match k_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing k_proj weight")),
        };
        let v_w = match v_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing v_proj weight")),
        };
        let o_w = match o_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing o_proj weight")),
        };
        let input_ln_w = match input_ln_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing input_layernorm weight")),
        };
        let post_attn_ln_w = match post_attn_ln_w {
            Some(w) => w,
            None => {
                return Err(candle_core::Error::msg(
                    "Missing post_attention_layernorm weight",
                ));
            }
        };

        let input_layernorm = Linear::new(input_ln_w, None);
        let post_attention_layernorm = Linear::new(post_attn_ln_w, None);

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            q_w,
            k_w,
            v_w,
            o_w,
            crate::components::AttentionConfig::default(),
            false,
            None,
            None,
        )?;

        let mut expert_weights = Vec::new();
        for i in 0..num_experts {
            let gate_w = weights
                .get(&format!(
                    "model.layers.{}.block_sparse_moe.experts.{}.gate_proj.weight",
                    layer_idx, i
                ))
                .cloned();
            let up_w = weights
                .get(&format!(
                    "model.layers.{}.block_sparse_moe.experts.{}.up_proj.weight",
                    layer_idx, i
                ))
                .cloned();
            let down_w = weights
                .get(&format!(
                    "model.layers.{}.block_sparse_moe.experts.{}.down_proj.weight",
                    layer_idx, i
                ))
                .cloned();

            let gate_w = match gate_w {
                Some(w) => w,
                None => {
                    return Err(candle_core::Error::msg(format!(
                        "Missing expert {}.gate_proj weight",
                        i
                    )));
                }
            };
            let up_w = match up_w {
                Some(w) => w,
                None => {
                    return Err(candle_core::Error::msg(format!(
                        "Missing expert {}.up_proj weight",
                        i
                    )));
                }
            };
            let down_w = match down_w {
                Some(w) => w,
                None => {
                    return Err(candle_core::Error::msg(format!(
                        "Missing expert {}.down_proj weight",
                        i
                    )));
                }
            };
            expert_weights.push((gate_w, up_w, down_w));
        }

        let gate_w = weights
            .get(&format!(
                "model.layers.{}.block_sparse_moe.gate.weight",
                layer_idx
            ))
            .cloned();
        let gate_w = match gate_w {
            Some(w) => w,
            None => return Err(candle_core::Error::msg("Missing gate weight")),
        };

        let mlp = MixtralSparseMoe::new_with_weights(
            hidden_size,
            num_experts,
            expert_intermediate_size,
            top_k,
            gate_w,
            expert_weights,
        )?;

        Ok(Self {
            attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            sliding_window,
        })
    }

    pub fn forward(&self, x: &Tensor, _positions: &[usize]) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.rms_norm(x, &self.input_layernorm)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.rms_norm(&x, &self.post_attention_layernorm)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Linear) -> Result<Tensor> {
        let hidden_size = x.dims().last().unwrap();
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        let total_len = batch_size * seq_len;
        let x_flat = x.reshape((total_len, *hidden_size))?;
        let weight = weight.weight().clone();
        let weight = weight.reshape((1, *hidden_size))?;

        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + 1e-6)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&weight)?;

        x.reshape((batch_size, seq_len, *hidden_size))
    }
}
