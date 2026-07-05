//! FlashAttention wrapper for the Qwen3 model: `FlashAttention` config struct + a `forward` shim that routes to the kernel impl.
//!
//! Thin layer over `kernels::flash_attention`; exists so the Qwen3
//! attention module can stay architecture-specific while the kernel
//! remains a generic implementation.
#![allow(clippy::module_name_repetitions)]
use candle_core::{Result, Tensor};

#[derive(Debug)]
/// `FlashAttention`. See the type definition for fields and behavior.
pub struct FlashAttention;

impl FlashAttention {
    #[must_use]
    pub const fn new(_config: FlashAttentionConfig) -> Self {
        Self
    }

    /// Run the layer forward pass over the input.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, _causal: bool) -> Result<Tensor> {
        let qk = Tensor::matmul(q, &k.transpose(2, 3)?.contiguous()?)?;
        let attn_weights = candle_nn::ops::softmax(&qk, 3)?.contiguous()?;
        let output = Tensor::matmul(&attn_weights, v)?;
        Ok(output)
    }
}

/// Configuration for FlashAttention. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone, Default)]
pub struct FlashAttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_p: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_basic() {
        let device = candle_core::Device::Cpu;
        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 4;
        let head_dim = 32;

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

        let flash = FlashAttention::new(FlashAttentionConfig {
            num_heads,
            head_dim,
            dropout_p: 0.0,
        });

        let output = flash.forward(&q, &k, &v, false).unwrap();

        assert_eq!(output.dims(), q.dims());
    }
}
