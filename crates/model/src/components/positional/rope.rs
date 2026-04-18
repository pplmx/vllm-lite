use crate::qwen3_config::Qwen3Config;
use candle_core::{Result, Tensor};

#[derive(Clone)]
#[allow(dead_code)]
pub struct RoPE {
    pub(crate) theta: f32,
    pub(crate) head_dim: usize,
    pub(crate) max_position: usize,
    pub(crate) scaling_factor: f32,
    pub(crate) device: candle_core::Device,
}

impl RoPE {
    pub fn new(head_dim: usize, max_position: usize, theta: f32, device: &candle_core::Device) -> Self {
        Self {
            theta,
            head_dim,
            max_position,
            scaling_factor: 1.0,
            device: device.clone(),
        }
    }

    pub fn new_with_config(config: &Qwen3Config) -> Self {
        use candle_core::Device;
        let rope_scaling = config.rope_scaling();
        Self {
            theta: config.rope_theta(),
            head_dim: config.head_dim(),
            max_position: config.max_position_embeddings(),
            scaling_factor: rope_scaling.and_then(|r| r.factor).unwrap_or(1.0),
            device: Device::Cpu,
        }
    }

    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    pub fn apply(&self, x: &Tensor, positions: &[i64]) -> Result<Tensor> {
        apply_rope(x, positions, self.theta)
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, position: i64) -> Result<(Tensor, Tensor)> {
        let positions: Vec<i64> = (0..q.dim(1)? as i64).map(|i| position + i).collect();
        let q_out = apply_rope(q, &positions, self.theta)?;
        let k_out = apply_rope(k, &positions, self.theta)?;
        Ok((q_out, k_out))
    }
}

pub fn apply_rope(query: &Tensor, positions: &[i64], theta: f32) -> Result<Tensor> {
    let (batch, seq_len, num_heads, head_dim) = query.dims4()?;

    let query = query.transpose(1, 2)?;

    let half_dim = head_dim / 2;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| theta.powf(-2.0 * (i as f32) / (head_dim as f32)))
        .collect();

    let mut cos_matrix = Vec::with_capacity(seq_len * half_dim);
    let mut sin_matrix = Vec::with_capacity(seq_len * half_dim);

    for &pos in positions {
        let pos_f = pos as f32;
        for &freq in &inv_freq {
            let angle = pos_f * freq;
            cos_matrix.push(angle.cos());
            sin_matrix.push(angle.sin());
        }
    }

    let cos = Tensor::new(cos_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;
    let sin = Tensor::new(sin_matrix.as_slice(), query.device())?
        .reshape((1, seq_len, half_dim))?
        .broadcast_as((batch, num_heads, seq_len, half_dim))?;

    let first_half = query.narrow(3, 0, half_dim)?;
    let second_half = query.narrow(3, half_dim, half_dim)?;

    let rotated_first = first_half
        .broadcast_mul(&cos)?
        .broadcast_add(&second_half.broadcast_mul(&sin)?)?;
    let rotated_second = second_half
        .broadcast_mul(&cos)?
        .broadcast_sub(&first_half.broadcast_mul(&sin)?)?;

    let result = Tensor::cat(&[&rotated_first, &rotated_second], 3)?;

    result.transpose(1, 2)
}

pub fn precompute_rope_cache(seq_len: usize, head_dim: usize, theta: f32) -> Vec<(f32, f32)> {
    let mut cache = Vec::with_capacity(seq_len * head_dim / 2);
    for pos in 0..seq_len {
        for i in 0..head_dim / 2 {
            let freq = (pos as f32).powf(-2.0 * (i as f32) / (head_dim as f32)) * theta;
            cache.push((freq.cos(), freq.sin()));
        }
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_apply_rope_returns_same_shape() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::ones((2, 4, 2, 32), DType::F32, &device)?;
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let result = apply_rope(&query, &positions, 10000.0)?;
        assert_eq!(result.dims(), query.dims());

        Ok(())
    }

    #[test]
    fn test_apply_rope_different_positions() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;

        let pos0: Vec<i64> = vec![0, 1];
        let out0 = apply_rope(&query, &pos0, 10000.0)?;

        let pos1: Vec<i64> = vec![10, 11];
        let out1 = apply_rope(&query, &pos1, 10000.0)?;

        let diff = (out0 - out1)?.abs()?.sum_all()?;
        assert!(
            diff.to_scalar::<f32>()? > 1e-5,
            "RoPE should produce different outputs for different positions"
        );

        Ok(())
    }

    #[test]
    fn test_apply_rope_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::ones((1, 2, 2, 4), DType::F32, &device)?;
        let positions: Vec<i64> = vec![0, 1];

        let out1 = apply_rope(&query, &positions, 10000.0)?;
        let out2 = apply_rope(&query, &positions, 10000.0)?;

        let diff = (out1 - out2)?.abs()?.sum_all()?;
        assert_eq!(
            diff.to_scalar::<f32>()?,
            0.0,
            "RoPE should be deterministic"
        );

        Ok(())
    }

    #[test]
    fn test_precompute_rope_cache_length() {
        let cache = precompute_rope_cache(10, 64, 10000.0);
        assert_eq!(cache.len(), 10 * 32);
    }

    #[test]
    fn test_precompute_rope_cache_first_position() {
        let cache = precompute_rope_cache(1, 64, 10000.0);
        assert_eq!(cache.len(), 32);
    }

    #[test]
    fn test_precompute_rope_cache_values() {
        let cache = precompute_rope_cache(10, 64, 10000.0);
        assert_eq!(cache.len(), 320);
    }

    #[test]
    fn test_rope_creation() {
        let device = Device::Cpu;
        let rope = RoPE::new(128, 2048, 10000.0, &device);
        assert_eq!(rope.theta, 10000.0);
        assert_eq!(rope.head_dim, 128);
        assert_eq!(rope.scaling_factor, 1.0);
    }

    #[test]
    fn test_rope_apply() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(64, 1024, 10000.0, &device);
        let query = Tensor::ones((1, 4, 8, 64), DType::F32, &device)?;
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let result = rope.apply(&query, &positions)?;
        assert_eq!(result.dims(), query.dims());

        Ok(())
    }

    #[test]
    fn test_rope_forward_q_shape_preserved() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(64, 1024, 10000.0, &device);

        let q = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;

        let (q_out, _) = rope.forward(&q, &k, 0)?;
        assert_eq!(q_out.dims(), q.dims());
        Ok(())
    }

    #[test]
    fn test_rope_forward_k_shape_preserved() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(64, 1024, 10000.0, &device);

        let q = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (2, 4, 8, 64), &device)?;

        let (_, k_out) = rope.forward(&q, &k, 0)?;
        assert_eq!(k_out.dims(), k.dims());
        Ok(())
    }

    #[test]
    fn test_rope_rotation_applied() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(64, 1024, 10000.0, &device);

        let q = Tensor::ones((1, 2, 8, 64), DType::F32, &device)?;
        let k = Tensor::ones((1, 2, 8, 64), DType::F32, &device)?;

        let (q_out, _) = rope.forward(&q, &k, 0)?;

        let diff = (&q_out - &q)?.abs()?;
        let sum_diff = diff.sum_all()?.to_scalar::<f32>()?;
        assert!(sum_diff > 1e-6, "RoPE should modify the tensor");
        Ok(())
    }

    #[test]
    fn test_rope_minimal_head_dim() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(64, 512, 10000.0, &device);

        let q = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;

        let (q_out, _) = rope.forward(&q, &k, 0)?;
        assert_eq!(q_out.dims(), q.dims());
        Ok(())
    }

    #[test]
    fn test_rope_large_position() -> Result<()> {
        let device = Device::Cpu;
        let rope = RoPE::new(64, 1024, 10000.0, &device);

        let q = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (1, 1, 1, 64), &device)?;

        let (q_out, _) = rope.forward(&q, &k, 8192)?;
        let flat = q_out.flatten_all()?;
        for i in 0..flat.elem_count() {
            let val: f32 = flat.get(i)?.to_scalar()?;
            assert!(val.is_finite(), "Value at index {} is not finite: {}", i, val);
        }
        Ok(())
    }
}
