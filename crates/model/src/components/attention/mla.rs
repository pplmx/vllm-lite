#![allow(clippy::too_many_arguments)]

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;
use tracing::trace;

use super::AttentionConfig;

pub struct MlaAttention {
    q_proj: Linear,
    kv_proj: Linear,
    k_decompress: Linear,
    v_decompress: Linear,
    o_proj: Linear,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_dim: usize,
    qk_rope_dim: usize,
    v_head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    config: AttentionConfig,
}

impl MlaAttention {
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn kv_lora_rank(&self) -> usize {
        self.kv_lora_rank
    }

    pub fn q_lora_rank(&self) -> usize {
        self.q_lora_rank
    }

    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    #[cfg(test)]
    pub fn q_proj_test(&self) -> &Linear {
        &self.q_proj
    }

    #[cfg(test)]
    pub fn kv_proj_test(&self) -> &Linear {
        &self.kv_proj
    }

    #[cfg(test)]
    pub fn k_decompress_test(&self) -> &Linear {
        &self.k_decompress
    }

    #[cfg(test)]
    pub fn v_decompress_test(&self) -> &Linear {
        &self.v_decompress
    }

    pub fn split_q(&self, q_compressed: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let batch_size = q_compressed.dims()[0];
        let q_nope_dim = self.num_heads * self.qk_nope_dim;
        let q_rope_dim_total = self.num_heads * self.qk_rope_dim;

        let q_reshaped =
            q_compressed.reshape((batch_size, seq_len, q_nope_dim + q_rope_dim_total))?;
        let q_nope = q_reshaped.narrow(2, 0, q_nope_dim)?;
        let q_rope = q_reshaped.narrow(2, q_nope_dim, q_rope_dim_total)?;

        Ok((q_nope, q_rope))
    }

    pub fn concat_q_nope_rope(&self, q_nope: &Tensor, q_rope: &Tensor) -> Result<Tensor> {
        let q = Tensor::cat(&[q_nope, q_rope], 2)?;
        let batch_size = q.dims()[0];
        let seq_len = q.dims()[1];
        let head_dim = self.qk_nope_dim + self.qk_rope_dim;
        let q = q.reshape((batch_size, seq_len, self.num_heads, head_dim))?;
        let q = q.transpose(1, 2)?;
        q.contiguous()
    }

    pub fn reshape_k(&self, k_flat: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let k = k_flat.reshape((batch_size, seq_len, self.num_kv_heads, self.v_head_dim))?;
        let k = k.transpose(1, 2)?;
        k.contiguous()
    }

    pub fn reshape_v(&self, v_flat: &Tensor, batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let v = v_flat.reshape((batch_size, seq_len, self.num_kv_heads, self.v_head_dim))?;
        let v = v.transpose(1, 2)?;
        v.contiguous()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_dim: usize,
        qk_rope_dim: usize,
        v_head_dim: usize,
        vb: Option<candle_nn::VarBuilder>,
        config: AttentionConfig,
    ) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let q_proj = candle_nn::linear(hidden_size, q_lora_rank, vb.pp("q_proj"))?;
        let kv_proj = candle_nn::linear(hidden_size, kv_lora_rank, vb.pp("kv_proj"))?;

        let k_decompress_out_dim = num_kv_heads * v_head_dim;
        let k_decompress = candle_nn::linear(kv_lora_rank, k_decompress_out_dim, vb.pp("k_decompress"))?;
        let v_decompress = candle_nn::linear(kv_lora_rank, k_decompress_out_dim, vb.pp("v_decompress"))?;

        let head_dim = qk_nope_dim + qk_rope_dim;
        let o_proj = candle_nn::linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            kv_proj,
            k_decompress,
            v_decompress,
            o_proj,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_attention_new_creation() {
        let attn = MlaAttention::new(
            2048,   // hidden_size
            16,     // num_heads
            16,     // num_kv_heads
            512,    // q_lora_rank
            512,    // kv_lora_rank
            128,    // qk_nope_dim
            64,     // qk_rope_dim
            128,    // v_head_dim
            None,   // vb
            AttentionConfig::default(),
        ).unwrap();

        assert_eq!(attn.num_heads(), 16);
        assert_eq!(attn.kv_lora_rank(), 512);
    }

    #[test]
    fn test_mla_attention_accessors() {
        let attn = MlaAttention::new(
            2048, 16, 16, 512, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        assert_eq!(attn.head_dim(), 128 + 64);  // qk_nope_dim + qk_rope_dim
        assert_eq!(attn.num_kv_heads(), 16);
        assert_eq!(attn.q_lora_rank(), 512);
    }

    #[test]
    fn test_mla_q_projection_shape() {
        let attn = MlaAttention::new(
            2048, 16, 16, 512, 512, 128, 64, 128, None, AttentionConfig::default(),
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
        let q_compressed = attn.q_proj_test().forward(&x).unwrap();

        assert_eq!(q_compressed.dims(), &[1, 4, 512]); // [batch, seq, q_lora_rank]
    }

    #[test]
    fn test_mla_split_q_shape() {
        let q_lora_rank = 16 * (128 + 64);
        let attn = MlaAttention::new(
            2048, 16, 16, q_lora_rank, 512, 128, 64, 128, None, AttentionConfig::default(),
        )
        .unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
        let q_compressed = attn.q_proj_test().forward(&x).unwrap();

        let (q_nope, q_rope) = attn.split_q(&q_compressed, 4).unwrap();

        // q_nope: [batch, seq, num_heads * qk_nope_dim] = [1, 4, 16 * 128]
        assert_eq!(q_nope.dims(), &[1, 4, 2048]);
        // q_rope: [batch, seq, num_heads * qk_rope_dim] = [1, 4, 16 * 64]
        assert_eq!(q_rope.dims(), &[1, 4, 1024]);
    }

    #[test]
    fn test_mla_kv_compression_shape() {
        let attn = MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &candle_core::Device::Cpu).unwrap();
        let kv_compressed = attn.kv_proj_test().forward(&x).unwrap();

        assert_eq!(kv_compressed.dims(), &[1, 4, 512]); // [batch, seq, kv_lora_rank]
    }

    #[test]
    fn test_mla_k_decompression_shape() {
        let attn = MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        let kv_compressed = Tensor::randn(0.0f32, 1.0, (1, 4, 512), &candle_core::Device::Cpu).unwrap();
        let k_decompressed = attn.k_decompress_test().forward(&kv_compressed).unwrap();

        // [batch, seq, num_kv_heads * v_head_dim] = [1, 4, 16 * 128]
        assert_eq!(k_decompressed.dims(), &[1, 4, 2048]);
    }

    #[test]
    fn test_mla_v_decompression_shape() {
        let attn = MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        let kv_compressed = Tensor::randn(0.0f32, 1.0, (1, 4, 512), &candle_core::Device::Cpu).unwrap();
        let v_decompressed = attn.v_decompress_test().forward(&kv_compressed).unwrap();

        // [batch, seq, num_kv_heads * v_head_dim] = [1, 4, 16 * 128]
        assert_eq!(v_decompressed.dims(), &[1, 4, 2048]);
    }

    #[test]
    fn test_mla_concat_q_nope_rope_shape() {
        let attn = MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        let batch_size = 1;
        let seq_len = 4;

        let q_nope = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, 16 * 128), &candle_core::Device::Cpu).unwrap();
        let q_rope = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, 16 * 64), &candle_core::Device::Cpu).unwrap();

        let q = attn.concat_q_nope_rope(&q_nope, &q_rope).unwrap();

        // Q: [batch, num_heads, seq, head_dim] = [1, 16, 4, 192]
        assert_eq!(q.dims(), &[1, 16, 4, 192]);
    }

    #[test]
    fn test_mla_rope_application() {
        use crate::components::positional::rope::apply_rope;

        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 16;
        let rope_dim = 64;

        let q_rope = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, num_heads, rope_dim), &candle_core::Device::Cpu).unwrap();
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let q_rope_rotated = apply_rope(&q_rope, &positions, 10000.0).unwrap();

        assert_eq!(q_rope_rotated.dims(), q_rope.dims());

        let diff = (&q_rope_rotated - &q_rope).unwrap().abs().unwrap();
        let sum_diff: f32 = diff.sum_all().unwrap().to_scalar().unwrap();
        assert!(sum_diff > 1e-5, "RoPE should modify the tensor");
    }

    #[test]
    fn test_mla_reshape_kv() {
        let attn = MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        let batch_size = 1;
        let seq_len = 4;
        let kv_compressed = Tensor::randn(0.0f32, 1.0, (batch_size, seq_len, 512), &candle_core::Device::Cpu).unwrap();

        let k_decompressed = attn.k_decompress_test().forward(&kv_compressed).unwrap();
        let k = attn.reshape_k(&k_decompressed, batch_size, seq_len).unwrap();

        // K: [batch, num_kv_heads, seq, v_head_dim] = [1, 16, 4, 128]
        assert_eq!(k.dims(), &[1, 16, 4, 128]);

        let v_decompressed = attn.v_decompress_test().forward(&kv_compressed).unwrap();
        let v = attn.reshape_v(&v_decompressed, batch_size, seq_len).unwrap();

        // V: [batch, num_kv_heads, seq, v_head_dim] = [1, 16, 4, 128]
        assert_eq!(v.dims(), &[1, 16, 4, 128]);
    }
}
