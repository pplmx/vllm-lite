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
}
