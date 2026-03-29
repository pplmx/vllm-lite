use super::{attention::GqaAttention, mlp::SwiGLU};
use candle_core::{Module, Result, Tensor};
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
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let input_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::layer_norm(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let vb_attn = vb.pp("attn");
        let attention = GqaAttention::new(hidden_size, num_heads, num_kv_heads, head_dim, vb_attn)?;

        let vb_mlp = vb.pp("mlp");
        let mlp = SwiGLU::new(hidden_size, intermediate_size, vb_mlp)?;

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
