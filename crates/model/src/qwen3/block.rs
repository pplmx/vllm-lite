#![allow(clippy::all, unused)]
use super::{attention::GqaAttention, mlp::SwiGLU};
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::LayerNorm;

pub struct TransformerBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GqaAttention,
    mlp: SwiGLU,
}

impl TransformerBlock {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        vb: Option<candle_nn::VarBuilder>,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let input_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let vb_attn = vb.pp("attn");
        let attention = GqaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            Some(vb_attn),
        )?;

        let vb_mlp = vb.pp("mlp");
        let mlp = SwiGLU::new(hidden_size, intermediate_size, Some(vb_mlp))?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    pub fn new_with_weights(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        weights: Option<(
            Option<Tensor>, // q_proj
            Option<Tensor>, // k_proj
            Option<Tensor>, // v_proj
            Option<Tensor>, // o_proj
            Option<Tensor>, // gate_proj
            Option<Tensor>, // up_proj
            Option<Tensor>, // down_proj
            Option<Tensor>, // input_layernorm
            Option<Tensor>, // post_attention_layernorm
        )>,
    ) -> Result<Self> {
        let Some((
            Some(q_w),
            Some(k_w),
            Some(v_w),
            Some(o_w),
            Some(gate_w),
            Some(up_w),
            Some(down_w),
            Some(input_ln_w),
            Some(post_attn_ln_w),
        )) = weights
        else {
            return Err(candle_core::Error::msg("Missing layer weights"));
        };

        let input_ln_dim = input_ln_w.dim(0).unwrap_or(hidden_size);
        let input_ln_bias = Tensor::zeros(input_ln_dim, input_ln_w.dtype(), &Device::Cpu)?;
        let input_layernorm = LayerNorm::new(input_ln_w.clone(), input_ln_bias, rms_norm_eps);

        let post_attn_dim = post_attn_ln_w.dim(0).unwrap_or(hidden_size);
        let post_attn_bias = Tensor::zeros(post_attn_dim, post_attn_ln_w.dtype(), &Device::Cpu)?;
        let post_attention_layernorm =
            LayerNorm::new(post_attn_ln_w.clone(), post_attn_bias, rms_norm_eps);

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
        )?;

        let mlp = SwiGLU::new_with_weights(hidden_size, intermediate_size, gate_w, up_w, down_w)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (x + residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x.add(&residual)
    }
}
