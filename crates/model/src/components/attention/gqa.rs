#![allow(clippy::too_many_arguments)]

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear};
use tracing::trace;

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

        trace!(
            batch_size,
            seq_len,
            head_dim = self.head_dim,
            "GqaAttention forward started"
        );

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

        trace!(output_shape = ?o.dims(), "GqaAttention forward completed");

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

    pub fn has_q_norm(&self) -> bool {
        self.q_norm.is_some()
    }

    pub fn has_k_norm(&self) -> bool {
        self.k_norm.is_some()
    }

    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    pub fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        Ok((q, k, v))
    }

    pub fn apply_q_norm_impl(
        &self,
        q: Tensor,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q.reshape((batch_size * num_heads * seq_len, head_dim))?;
            let q = q_norm.forward(&q)?;
            let q = q.reshape((batch_size, num_heads, seq_len, head_dim))?;
            Ok(q)
        } else {
            Ok(q)
        }
    }

    pub fn apply_k_norm_impl(
        &self,
        k: Tensor,
        batch_size: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k.reshape((batch_size * num_kv_heads * seq_len, head_dim))?;
            let k = k_norm.forward(&k)?;
            let k = k.reshape((batch_size, num_kv_heads, seq_len, head_dim))?;
            Ok(k)
        } else {
            Ok(k)
        }
    }

    pub fn apply_q_norm_impl_flattened(&self, q: Tensor) -> Result<Tensor> {
        if let Some(ref q_norm) = self.q_norm {
            let q = q_norm.forward(&q)?;
            Ok(q)
        } else {
            Ok(q)
        }
    }

    pub fn apply_k_norm_impl_flattened(&self, k: Tensor) -> Result<Tensor> {
        if let Some(ref k_norm) = self.k_norm {
            let k = k_norm.forward(&k)?;
            Ok(k)
        } else {
            Ok(k)
        }
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

    #[test]
    fn test_gqa_attention_new_creation() -> Result<()> {
        let _device = candle_core::Device::Cpu;
        let attn = GqaAttention::new(256, 8, 2, 32, None, AttentionConfig::default(), false)?;
        assert_eq!(attn.num_heads(), 8);
        assert_eq!(attn.num_kv_heads(), 2);
        assert_eq!(attn.head_dim(), 32);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_num_heads_accessors() {
        let _device = candle_core::Device::Cpu;
        let attn =
            GqaAttention::new(512, 16, 4, 32, None, AttentionConfig::default(), false).unwrap();
        assert_eq!(attn.num_heads(), 16);
        assert_eq!(attn.num_kv_heads(), 4);
    }

    #[test]
    fn test_gqa_attention_head_dim_accessors() {
        let _device = candle_core::Device::Cpu;
        let attn =
            GqaAttention::new(256, 4, 2, 64, None, AttentionConfig::default(), false).unwrap();
        assert_eq!(attn.head_dim(), 64);
    }

    #[test]
    fn test_gqa_attention_paged_attention_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 64;
        let batch_size = 2;
        let seq_len = 4;
        let hidden_size = num_heads * head_dim;

        let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
        let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
        let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
        let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;

        let attn = GqaAttention::new_with_weights(
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
        )?;

        let q = Tensor::randn(0.0f32, 1.0, (batch_size, num_heads, 1, head_dim), &device)?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let output = attn.paged_attention_fn(&q, &k, &v)?;
        assert_eq!(output.dims(), &[batch_size, 1, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_tiled_attention_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 32;
        let batch_size = 1;
        let seq_len = 8;
        let hidden_size = num_heads * head_dim;

        let q_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;
        let k_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
        let v_w = Tensor::randn(0.0f32, 1.0, (num_kv_heads * head_dim, hidden_size), &device)?;
        let o_w = Tensor::randn(0.0f32, 1.0, (hidden_size, hidden_size), &device)?;

        let attn = GqaAttention::new_with_weights(
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
        )?;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            &device,
        )?;

        let output = attn.tiled_attention_fn(&q, &k, &v)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_single_q_head() -> Result<()> {
        let _device = candle_core::Device::Cpu;
        let attn = GqaAttention::new(64, 1, 1, 64, None, AttentionConfig::default(), false)?;
        assert_eq!(attn.num_heads(), 1);
        assert_eq!(attn.num_kv_heads(), 1);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_matching_heads() -> Result<()> {
        let _device = candle_core::Device::Cpu;
        let attn = GqaAttention::new(256, 4, 4, 64, None, AttentionConfig::default(), false)?;
        assert_eq!(attn.num_heads(), attn.num_kv_heads());
        Ok(())
    }

    #[test]
    fn test_gqa_attention_small_head_dim() -> Result<()> {
        let _device = candle_core::Device::Cpu;
        let attn = GqaAttention::new(64, 2, 1, 32, None, AttentionConfig::default(), false)?;
        assert_eq!(attn.head_dim(), 32);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_large_head_dim() -> Result<()> {
        let _device = candle_core::Device::Cpu;
        let attn = GqaAttention::new(512, 4, 2, 128, None, AttentionConfig::default(), false)?;
        assert_eq!(attn.head_dim(), 128);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_output_finite() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let attn = GqaAttention::new(128, 4, 2, 32, None, AttentionConfig::default(), false)?;

        let q = Tensor::randn(-2.0f32, 2.0, (1, 4, 4, 32), &device)?;
        let k = Tensor::randn(-2.0f32, 2.0, (1, 4, 4, 32), &device)?;
        let v = Tensor::randn(-2.0f32, 2.0, (1, 4, 4, 32), &device)?;

        let output = attn.paged_attention_fn(&q, &k, &v)?;
        let data: Vec<f32> = output.flatten_all()?.to_vec1()?;
        assert!(data.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn test_gqa_attention_deterministic() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let q_w = Tensor::randn(0.0f32, 1.0, (256, 256), &device)?;
        let k_w = Tensor::randn(0.0f32, 1.0, (64, 256), &device)?;
        let v_w = Tensor::randn(0.0f32, 1.0, (64, 256), &device)?;
        let o_w = Tensor::randn(0.0f32, 1.0, (256, 256), &device)?;

        let attn = GqaAttention::new_with_weights(
            256,
            8,
            2,
            32,
            q_w,
            k_w,
            v_w,
            o_w,
            AttentionConfig::default(),
            false,
            None,
            None,
        )?;

        let q = Tensor::randn(0.0f32, 1.0, (1, 8, 1, 32), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 32), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 32), &device)?;

        let out1 = attn.paged_attention_fn(&q, &k, &v)?;
        let out2 = attn.paged_attention_fn(&q, &k, &v)?;

        let diff = (&out1 - &out2)?.abs()?;
        let max_diff: f32 = diff
            .flatten_all()?
            .to_vec1()?
            .iter()
            .cloned()
            .fold(0.0f32, |a, b| a.max(b));
        assert!(max_diff < 1e-5);
        Ok(())
    }

    #[test]
    fn test_gqa_attention_expand_kv_correct() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let _attn = GqaAttention::new(256, 8, 2, 32, None, AttentionConfig::default(), false)?;

        let num_kv_heads = 2;
        let head_dim = 32;
        let seq_len = 4;

        let k = Tensor::randn(0.0f32, 1.0, (1, seq_len, num_kv_heads, head_dim), &device)?;
        let k_expanded = expand_kv(&k, 8, 2)?;

        assert_eq!(k_expanded.dims(), &[1, seq_len, 8, head_dim]);
        Ok(())
    }
}
