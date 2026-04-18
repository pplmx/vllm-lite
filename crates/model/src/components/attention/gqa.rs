#![allow(clippy::too_many_arguments)]

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};

use super::{AttentionConfig, expand_kv, paged_attention, tiled_attention};

pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
}

impl GqaAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vb: Option<candle_nn::VarBuilder>,
        config: AttentionConfig,
        has_qk_norm: bool,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let q_proj = candle_nn::linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = if has_qk_norm {
            Some(candle_nn::layer_norm(head_dim, 1e-6, vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            Some(candle_nn::layer_norm(head_dim, 1e-6, vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            config,
            q_norm,
            k_norm,
        })
    }

    pub fn new_with_weights(
        _hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        config: AttentionConfig,
        has_qk_norm: bool,
        q_norm_weight: Option<Tensor>,
        k_norm_weight: Option<Tensor>,
    ) -> Result<Self> {
        let q_proj = Linear::new(q_weight, None);
        let k_proj = Linear::new(k_weight, None);
        let v_proj = Linear::new(v_weight, None);
        let o_proj = Linear::new(o_weight, None);

        let q_norm = if has_qk_norm {
            let q_norm_weight =
                q_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing q_norm weight"))?;
            let q_norm_bias =
                Tensor::zeros(head_dim, q_norm_weight.dtype(), q_norm_weight.device())?;
            Some(LayerNorm::new(q_norm_weight, q_norm_bias, 1e-6))
        } else {
            None
        };
        let k_norm = if has_qk_norm {
            let k_norm_weight =
                k_norm_weight.ok_or_else(|| candle_core::Error::msg("Missing k_norm weight"))?;
            let k_norm_bias =
                Tensor::zeros(head_dim, k_norm_weight.dtype(), k_norm_weight.device())?;
            Some(LayerNorm::new(k_norm_weight, k_norm_bias, 1e-6))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            config,
            q_norm,
            k_norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let seq_len = x.dims()[1];

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;

        let q = self.apply_q_norm(q, batch_size, seq_len)?;
        let k = self.apply_k_norm(k, batch_size, seq_len)?;

        let k = self.expand_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = self.expand_kv(&v, self.num_heads, self.num_kv_heads)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = q.contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let qk = Tensor::matmul(&q, &k_t)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?.contiguous()?;

        let attn_output = Tensor::matmul(&attn_weights, &v.contiguous()?)?;

        let attn_output = attn_output.transpose(1, 2)?;

        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    pub fn expand_kv(
        &self,
        kv: &Tensor,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Tensor> {
        expand_kv(kv, num_q_heads, num_kv_heads)
    }

    pub fn paged_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let attn_output = paged_attention(q, k, v, self.num_heads, self.head_dim)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    pub fn tiled_attention_fn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let tile_size = self.config.tile_size.unwrap_or(16);
        let attn_output = tiled_attention(q, k, v, self.num_heads, tile_size)?;
        let o = self.o_proj.forward(&attn_output)?;
        Ok(o)
    }

    fn apply_q_norm(&self, q: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q.transpose(1, 2)?;
            let reshape_size = batch_size * self.num_heads * seq_len;
            let q = q.reshape((reshape_size, self.head_dim))?;
            let q = q_norm.forward(&q)?;
            let q = q.reshape((batch_size, self.num_heads, seq_len, self.head_dim))?;
            let q = q.transpose(1, 2)?;
            Ok(q)
        } else {
            Ok(q)
        }
    }

    fn apply_k_norm(&self, k: Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k.transpose(1, 2)?;
            let reshape_size = batch_size * self.num_kv_heads * seq_len;
            let k = k.reshape((reshape_size, self.head_dim))?;
            let k = k_norm.forward(&k)?;
            let k = k.reshape((batch_size, self.num_kv_heads, seq_len, self.head_dim))?;
            let k = k.transpose(1, 2)?;
            Ok(k)
        } else {
            Ok(k)
        }
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_attention_forward_output_shape() {
        let device = candle_core::Device::Cpu;
        let num_heads = 16;
        let num_kv_heads = 8;
        let head_dim = 128;
        let batch_size = 1;
        let seq_len = 8;

        let hidden_size = num_heads * head_dim;
        let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
        let k_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let v_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            false,
            None,
            None,
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
        let output = attention.forward(&x).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_gqa_attention_with_qk_norm() {
        let device = candle_core::Device::Cpu;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;
        let batch_size = 1;
        let seq_len = 4;

        let hidden_size = num_heads * head_dim;
        let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
        let k_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let v_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

        let q_norm_w = Tensor::randn(0.0f32, 0.1, (head_dim,), &device).unwrap();
        let k_norm_w = Tensor::randn(0.0f32, 0.1, (head_dim,), &device).unwrap();

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            true,
            Some(q_norm_w),
            Some(k_norm_w),
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
        let output = attention.forward(&x).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_gqa_attention_paged_attention_output_shape() {
        let device = candle_core::Device::Cpu;
        let num_heads = 16;
        let num_kv_heads = 8;
        let head_dim = 128;
        let batch_size = 1;

        let hidden_size = num_heads * head_dim;
        let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
        let k_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let v_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            false,
            None,
            None,
        )
        .unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 1, head_dim), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 8, head_dim), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 8, head_dim), &device).unwrap();

        let output = attention.paged_attention_fn(&q, &k, &v).unwrap();

        assert_eq!(output.dims(), &[batch_size, 1, hidden_size]);
    }

    #[test]
    fn test_gqa_attention_tiled_attention_output_shape() {
        let device = candle_core::Device::Cpu;
        let num_heads = 16;
        let num_kv_heads = 8;
        let head_dim = 128;
        let batch_size = 1;
        let seq_len = 8;

        let hidden_size = num_heads * head_dim;
        let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();
        let k_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let v_w =
            Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device).unwrap();
        let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device).unwrap();

        let attention = GqaAttention::new_with_weights(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            false,
            None,
            None,
        )
        .unwrap();

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )
        .unwrap();

        let output = attention.tiled_attention_fn(&q, &k, &v).unwrap();

        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }
}
