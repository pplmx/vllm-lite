use crate::components::AttentionConfig;
use crate::components::MlaAttention;
use candle_core::{Result, Tensor};

pub struct Qwen3MlaAttention {
    inner: MlaAttention,
}

impl Qwen3MlaAttention {
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
        let inner = MlaAttention::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            vb,
            config,
        )?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        self.inner.forward(x, positions)
    }

    pub fn inner(&self) -> &MlaAttention {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_mla_attention_creation() {
        let attn = Qwen3MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        assert_eq!(attn.inner().num_heads(), 16);
        assert_eq!(attn.inner().kv_lora_rank(), 512);
    }

    #[test]
    fn test_qwen3_mla_attention_forward() {
        use candle_core::Device;

        let attn = Qwen3MlaAttention::new(
            2048, 16, 16, 3072, 512, 128, 64, 128, None, AttentionConfig::default()
        ).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 2048), &Device::Cpu).unwrap();
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let output = attn.forward(&x, &positions).unwrap();
        assert_eq!(output.dims(), &[1, 4, 2048]);
    }
}
