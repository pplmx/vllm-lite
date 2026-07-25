#![allow(clippy::module_name_repetitions)]
//! Fused attention layer kernel.

// invariant: tensor-dimension casts (head_dim -> f32) are bounded by model
// architecture constants; precision loss is intentional.
#![allow(clippy::cast_precision_loss)]

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Fused attention layer: layernorm + attention + residual
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_layer(
    x: &Tensor,
    layernorm_weight: &Tensor,
    _layernorm_bias: &Tensor,
    q_proj: &Linear,
    k_proj: &Linear,
    v_proj: &Linear,
    o_proj: &Linear,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
) -> Result<Tensor> {
    // 1. RMS Norm
    let (batch, seq, hidden) = x.dims3()?;
    let flat_size = batch * seq;
    let x_flat = x.reshape((flat_size, hidden))?;
    let weight_2d = layernorm_weight.reshape((1, hidden))?;
    let variance = x_flat.sqr()?.mean_keepdim(1)?;
    let x_normed = x_flat.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let x_normed = x_normed.broadcast_mul(&weight_2d)?;
    let x = x_normed.reshape((batch, seq, hidden))?;

    // 2. QKV projection
    let q = q_proj.forward(&x)?;
    let k = k_proj.forward(&x)?;
    let v = v_proj.forward(&x)?;

    // 3. Reshape for attention
    // `contiguous()` after transpose is required: CUDA matmul rejects
    // non-contiguous LHS/RHS (MatMulUnexpectedStriding). See GQA/forward.rs
    // (H-11 #3). On CPU the cost is a single strided copy at (B, H, S, D).
    let q = q
        .reshape((batch, seq, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = k
        .reshape((batch, seq, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = v
        .reshape((batch, seq, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;

    // 4. Simple attention (no causal mask for now)
    // `affine(scale, 0.0)` fuses the scaling into the matmul kernel without
    // materializing a broadcast tensor — see GQA/forward.rs (H-11 #2).
    let scale = 1.0f64 / (head_dim as f64).sqrt();
    let qk = Tensor::matmul(&q, &k.transpose(2, 3)?.contiguous()?)?;
    let qk = qk.affine(scale, 0.0)?;
    let attn = candle_nn::ops::softmax(&qk, 3)?;
    let out = Tensor::matmul(&attn, &v)?;

    // 5. Reshape and output projection
    let out = out
        .transpose(1, 2)?
        .reshape((batch, seq, num_heads * head_dim))?;
    o_proj.forward(&out)
}

/// # Errors
///
/// Returns `Err` if the operation fails.
/// Fused MLP layer: layernorm + `gate_proj` + `up_proj` + `down_proj` + residual
pub fn fused_mlp_layer(
    x: &Tensor,
    layernorm_weight: &Tensor,
    gate_proj: &Linear,
    up_proj: &Linear,
    down_proj: &Linear,
    eps: f64,
) -> Result<Tensor> {
    // 1. RMS Norm
    let dims = x.dims();
    let hidden = *dims
        .last()
        .ok_or_else(|| candle_core::Error::msg("input tensor has no dimensions"))?;
    let flat_size: usize = dims[..dims.len() - 1].iter().product();
    let x_flat = x.reshape((flat_size, hidden))?;
    let weight_2d = layernorm_weight.reshape((1, hidden))?;
    let variance = x_flat.sqr()?.mean_keepdim(1)?;
    let x_normed = x_flat.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let x_normed = x_normed.broadcast_mul(&weight_2d)?;
    let x = x_normed.reshape(dims)?;

    // 2. SwiGLU: gate * up -> silu -> * down
    let gate = gate_proj.forward(&x)?;
    let up = up_proj.forward(&x)?;
    let activated = gate.silu()?.broadcast_mul(&up)?;
    down_proj.forward(&activated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::Linear;

    /// Build a random `Linear(in, out)` with no bias (matches the fused-kernel
    /// call sites which pass bias-less projections).
    fn random_linear(in_dim: usize, out_dim: usize, device: &Device) -> Linear {
        let weight = Tensor::randn(0.0f32, 0.05, (out_dim, in_dim), device).unwrap();
        Linear::new(weight, None)
    }

    /// Device used for CPU tests — always `Cpu`.
    fn cpu_device() -> Device {
        Device::Cpu
    }

    /// Device used for CUDA tests — `Cuda(0)` when the `cuda` feature is enabled
    /// AND a GPU is present; falls back to `Cpu` otherwise.
    fn gpu_or_cpu_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            if let Ok(dev) = Device::cuda_if_available(0) {
                return dev;
            }
        }
        Device::Cpu
    }

    #[test]
    fn fused_attention_layer_output_shape() -> Result<()> {
        let device = cpu_device();
        let batch = 1;
        let seq = 8;
        let hidden = 256;
        let num_heads = 4;
        let head_dim = 32; // num_heads * head_dim = 128 (QKV proj dim)

        let x = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), &device)?;
        let ln_weight = Tensor::randn(0.0f32, 1.0, (hidden,), &device)?;
        let ln_bias = Tensor::zeros((hidden,), DType::F32, &device)?;

        let q_proj = random_linear(hidden, num_heads * head_dim, &device);
        let k_proj = random_linear(hidden, num_heads * head_dim, &device);
        let v_proj = random_linear(hidden, num_heads * head_dim, &device);
        let o_proj = random_linear(num_heads * head_dim, hidden, &device);

        let out = fused_attention_layer(
            &x, &ln_weight, &ln_bias, &q_proj, &k_proj, &v_proj, &o_proj, num_heads, head_dim, 1e-6,
        )?;

        assert_eq!(out.dims(), &[batch, seq, hidden]);
        Ok(())
    }

    #[test]
    fn fused_mlp_layer_output_shape() -> Result<()> {
        let device = cpu_device();
        let batch = 1;
        let seq = 8;
        let hidden = 256;
        let intermediate = 512;

        let x = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), &device)?;
        let ln_weight = Tensor::randn(0.0f32, 1.0, (hidden,), &device)?;

        let gate_proj = random_linear(hidden, intermediate, &device);
        let up_proj = random_linear(hidden, intermediate, &device);
        let down_proj = random_linear(intermediate, hidden, &device);

        let out = fused_mlp_layer(&x, &ln_weight, &gate_proj, &up_proj, &down_proj, 1e-6)?;

        assert_eq!(out.dims(), &[batch, seq, hidden]);
        Ok(())
    }

    // ── CUDA / GPU-enabled test variants ───────────────────────────
    // These tests run on GPU when the `cuda` feature is enabled and a GPU is
    // available. They verify the kernels produce correct output shapes on the
    // CUDA device. When no GPU is present they silently fall back to CPU.

    #[test]
    fn fused_attention_layer_cuda() -> Result<()> {
        let device = gpu_or_cpu_device();
        let batch = 2;
        let seq = 16;
        let hidden = 512;
        let num_heads = 8;
        let head_dim = 64;

        let x = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), &device)?;
        let ln_weight = Tensor::randn(0.0f32, 1.0, (hidden,), &device)?;
        let ln_bias = Tensor::zeros((hidden,), DType::F32, &device)?;

        let q_proj = random_linear(hidden, num_heads * head_dim, &device);
        let k_proj = random_linear(hidden, num_heads * head_dim, &device);
        let v_proj = random_linear(hidden, num_heads * head_dim, &device);
        let o_proj = random_linear(num_heads * head_dim, hidden, &device);

        let out = fused_attention_layer(
            &x, &ln_weight, &ln_bias, &q_proj, &k_proj, &v_proj, &o_proj, num_heads, head_dim, 1e-6,
        )?;

        assert_eq!(out.dims(), &[batch, seq, hidden]);
        // Verify the output is on the same device as the input.
        assert!(device_device_match(&out, &device));
        Ok(())
    }

    #[test]
    fn fused_mlp_layer_cuda() -> Result<()> {
        let device = gpu_or_cpu_device();
        let batch = 2;
        let seq = 16;
        let hidden = 512;
        let intermediate = 1024;

        let x = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), &device)?;
        let ln_weight = Tensor::randn(0.0f32, 1.0, (hidden,), &device)?;

        let gate_proj = random_linear(hidden, intermediate, &device);
        let up_proj = random_linear(hidden, intermediate, &device);
        let down_proj = random_linear(intermediate, hidden, &device);

        let out = fused_mlp_layer(&x, &ln_weight, &gate_proj, &up_proj, &down_proj, 1e-6)?;

        assert_eq!(out.dims(), &[batch, seq, hidden]);
        assert!(device_device_match(&out, &device));
        Ok(())
    }

    #[test]
    #[cfg_attr(not(feature = "cuda"), ignore = "requires the 'cuda' feature")]
    fn fused_attention_layer_multi_batch_cuda() -> Result<()> {
        let device = device_only_cuda();
        // Multi-batch: 4 sequences of varying conceptual lengths in one batch.
        let batch = 4;
        let seq = 32;
        let hidden = 768;
        let num_heads = 6;
        let head_dim = 128;

        let x = Tensor::randn(0.0f32, 0.5, (batch, seq, hidden), &device)?;
        let ln_weight = Tensor::randn(0.0f32, 1.0, (hidden,), &device)?;
        let ln_bias = Tensor::zeros((hidden,), DType::F32, &device)?;

        let q_proj = random_linear(hidden, num_heads * head_dim, &device);
        let k_proj = random_linear(hidden, num_heads * head_dim, &device);
        let v_proj = random_linear(hidden, num_heads * head_dim, &device);
        let o_proj = random_linear(num_heads * head_dim, hidden, &device);

        let out = fused_attention_layer(
            &x, &ln_weight, &ln_bias, &q_proj, &k_proj, &v_proj, &o_proj, num_heads, head_dim, 1e-6,
        )?;

        assert_eq!(out.dims(), &[batch, seq, hidden]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn device_only_cuda() -> Device {
        let device =
            Device::cuda_if_available(0).expect("CUDA device must be available for this test");
        assert!(
            matches!(device, Device::Cuda(_)),
            "expected CUDA device, got {device:?}"
        );
        device
    }

    #[cfg(not(feature = "cuda"))]
    fn device_only_cuda() -> Device {
        Device::Cpu
    }

    /// Check if a tensor is on the same device type as the given device.
    /// `Device` doesn't implement `PartialEq`, so we compare by variant.
    fn device_device_match(tensor: &Tensor, device: &Device) -> bool {
        matches!(
            (tensor.device(), device),
            (Device::Cpu, Device::Cpu) | (Device::Cuda(_), Device::Cuda(_))
        )
    }
}
