//! Gemma4 Rotary Position Embedding implementation.

use crate::config::architecture::RoPEConfig;
use candle_core::{DType, Result, Tensor};

pub struct Gemma4RoPE {
    rope_theta: f32,
    partial_rotary_factor: f32,
    head_dim: usize,
}

impl Gemma4RoPE {
    pub fn new(rope_config: &RoPEConfig, head_dim: usize) -> Self {
        Self {
            rope_theta: rope_config.rope_theta,
            partial_rotary_factor: rope_config.partial_rotary_factor,
            head_dim,
        }
    }

    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    fn apply_single(&self, x: &Tensor, positions: &[i64], _is_query: bool) -> Result<Tensor> {
        let rot_dim = self.rot_dim();
        if rot_dim == 0 {
            return Ok(x.clone());
        }

        let (batch, num_heads, seq_len, head_dim) = x.dims4()?;
        let half_rot_dim = rot_dim / 2;

        let inv_freq: Vec<f32> = (0..half_rot_dim)
            .map(|i| self.rope_theta.powf(-2.0 * (i as f32) / (rot_dim as f32)))
            .collect();

        let mut cos_matrix = Vec::with_capacity(seq_len * half_rot_dim);
        let mut sin_matrix = Vec::with_capacity(seq_len * half_rot_dim);

        for &pos in positions {
            let pos_f = pos as f32;
            for &freq in &inv_freq {
                let angle = pos_f * freq;
                cos_matrix.push(angle.cos());
                sin_matrix.push(angle.sin());
            }
        }

        let cos = Tensor::new(cos_matrix.as_slice(), x.device())?
            .reshape((seq_len, half_rot_dim))?
            .broadcast_add(&Tensor::zeros(
                (batch, num_heads, 1, 1),
                DType::F32,
                x.device(),
            )?)?
            .reshape((1, batch * num_heads, seq_len, half_rot_dim))?
            .squeeze(0)?;
        let sin = Tensor::new(sin_matrix.as_slice(), x.device())?
            .reshape((seq_len, half_rot_dim))?
            .broadcast_add(&Tensor::zeros(
                (batch, num_heads, 1, 1),
                DType::F32,
                x.device(),
            )?)?
            .reshape((1, batch * num_heads, seq_len, half_rot_dim))?
            .squeeze(0)?;

        let x_rot = x.reshape((batch * num_heads, seq_len, head_dim))?;
        let x_first = x_rot.narrow(2, 0, half_rot_dim)?;
        let x_second = x_rot.narrow(2, half_rot_dim, half_rot_dim)?;

        let rotated_first = x_first
            .broadcast_mul(&cos)?
            .broadcast_add(&x_second.broadcast_mul(&sin)?)?;
        let rotated_second = x_second
            .broadcast_mul(&cos)?
            .broadcast_sub(&x_first.broadcast_mul(&sin)?)?;

        let x_rotated = Tensor::cat(&[&rotated_first, &rotated_second], 2)?;

        let x_out = if head_dim > rot_dim {
            let x_remainder = x_rot.narrow(2, rot_dim, head_dim - rot_dim)?;
            Tensor::cat(&[&x_rotated, &x_remainder], 2)?
        } else {
            x_rotated
        };

        x_out.reshape((batch, num_heads, seq_len, head_dim))
    }

    /// Apply p-RoPE (partial Rotary Position Embedding) to query and key tensors
    /// Only the first `rot_dim` dimensions are rotated, the rest remain unchanged
    pub fn apply(&self, q: &Tensor, k: &Tensor, positions: &[i64]) -> Result<(Tensor, Tensor)> {
        let q_out = self.apply_single(q, positions, true)?;
        let k_out = self.apply_single(k, positions, false)?;
        Ok((q_out, k_out))
    }
}

impl Default for Gemma4RoPE {
    fn default() -> Self {
        Self {
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            head_dim: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_rope_config_creation() {
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
        };
        let rope = Gemma4RoPE::new(&config, 256);

        assert_eq!(rope.rope_theta, 10000.0);
        assert_eq!(rope.partial_rotary_factor, 1.0);
        assert_eq!(rope.head_dim, 256);
    }

    #[test]
    fn test_rope_config_full_attention() {
        let config = RoPEConfig {
            rope_theta: 1000000.0,
            partial_rotary_factor: 0.25,
        };
        let rope = Gemma4RoPE::new(&config, 256);

        assert_eq!(rope.rope_theta, 1000000.0);
        assert_eq!(rope.partial_rotary_factor, 0.25);
        assert_eq!(rope.rot_dim(), 64);
    }

    #[test]
    fn test_rope_apply_returns_same_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
        };
        let rope = Gemma4RoPE::new(&config, 256);

        let q = Tensor::ones((1, 8, 10, 256), DType::F32, &device)?;
        let k = Tensor::ones((1, 8, 10, 256), DType::F32, &device)?;

        let (q_out, k_out) = rope.apply(&q, &k, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])?;

        assert_eq!(q_out.dims(), q.dims());
        assert_eq!(k_out.dims(), k.dims());
        Ok(())
    }

    #[test]
    fn test_rope_default() {
        let rope = Gemma4RoPE::default();
        assert_eq!(rope.rope_theta, 10000.0);
        assert_eq!(rope.partial_rotary_factor, 0.5);
    }

    #[test]
    fn test_partial_rope_rot_dim() {
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
        };
        let rope = Gemma4RoPE::new(&config, 256);
        assert_eq!(rope.rot_dim(), 128);
    }

    #[test]
    fn test_partial_rope_unchanged_dims() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
        };
        let rope = Gemma4RoPE::new(&config, 256);

        let q = Tensor::zeros((1, 2, 4, 256), DType::F32, &device)?;
        let k = Tensor::zeros((1, 2, 4, 256), DType::F32, &device)?;

        let positions: Vec<i64> = vec![0, 1, 2, 3];
        let (q_out, k_out) = rope.apply(&q, &k, &positions)?;

        let q_remainder = q_out.narrow(3, 128, 128)?;
        let k_remainder = k_out.narrow(3, 128, 128)?;

        let q_rem_sum = q_remainder.sum_all()?.to_scalar::<f32>()?;
        let k_rem_sum = k_remainder.sum_all()?.to_scalar::<f32>()?;

        assert_eq!(q_rem_sum, 0.0);
        assert_eq!(k_rem_sum, 0.0);
        Ok(())
    }

    #[test]
    fn test_rope_different_positions() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
        };
        let rope = Gemma4RoPE::new(&config, 64);

        let q = Tensor::ones((1, 1, 1, 64), DType::F32, &device)?;

        let (q_out0, _) = rope.apply(&q, &q, &[0])?;
        let (q_out10, _) = rope.apply(&q, &q, &[10])?;

        let diff = (q_out0 - q_out10)?.abs()?.sum_all()?;
        assert!(
            diff.to_scalar::<f32>()? > 1e-5,
            "RoPE should produce different outputs for different positions"
        );
        Ok(())
    }

    #[test]
    fn test_rope_rot_dim_zero() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 0.0,
        };
        let rope = Gemma4RoPE::new(&config, 256);

        let q = Tensor::ones((1, 2, 4, 256), DType::F32, &device)?;
        let k = Tensor::ones((1, 2, 4, 256), DType::F32, &device)?;

        let positions: Vec<i64> = vec![0, 1, 2, 3];
        let (q_out, k_out) = rope.apply(&q, &k, &positions)?;

        let diff_q = (q_out - q)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        let diff_k = (k_out - k)?.abs()?.sum_all()?.to_scalar::<f32>()?;

        assert_eq!(diff_q, 0.0);
        assert_eq!(diff_k, 0.0);
        Ok(())
    }

    #[test]
    fn test_rope_single_head() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let config = RoPEConfig {
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
        };
        let rope = Gemma4RoPE::new(&config, 128);

        let q = Tensor::ones((2, 1, 5, 128), DType::F32, &device)?;
        let k = Tensor::ones((2, 1, 5, 128), DType::F32, &device)?;

        let positions: Vec<i64> = vec![0, 1, 2, 3, 4];
        let (q_out, k_out) = rope.apply(&q, &k, &positions)?;

        assert_eq!(q_out.dims(), q.dims());
        assert_eq!(k_out.dims(), k.dims());
        Ok(())
    }
}
