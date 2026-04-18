//! RMSNorm (Root Mean Square Layer Normalization) implementation.

use candle_core::{Result, Tensor};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();

        if dims.len() == 3 {
            let (batch, seq, hidden) = x.dims3()?;
            let x_flat = x.reshape((batch * seq, hidden))?;
            let weight_2d = self.weight.reshape((1, hidden))?;

            let variance = x_flat.sqr()?.mean_keepdim(1)?;
            let x_normed = x_flat.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            let x = x_normed.broadcast_mul(&weight_2d)?;

            x.reshape((batch, seq, hidden))
        } else {
            let hidden = *dims
                .last()
                .ok_or_else(|| candle_core::Error::msg("Empty tensor"))?;
            let weight_2d = self.weight.reshape((1, hidden))?;

            let variance = x.sqr()?.mean_keepdim(1)?;
            let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            x_normed.broadcast_mul(&weight_2d)
        }
    }
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dims = x.dims();

    if dims.len() == 3 {
        let (batch, seq, hidden) = x.dims3()?;
        let x_flat = x.reshape((batch * seq, hidden))?;
        let weight_2d = weight.reshape((1, hidden))?;

        let variance = x_flat.sqr()?.mean(1)?;
        let x_normed = x_flat.broadcast_div(&(variance + eps)?.sqrt()?)?;
        let x = x_normed.broadcast_mul(&weight_2d)?;

        x.reshape((batch, seq, hidden))
    } else if dims.len() == 2 {
        let hidden = dims[1];

        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(1)?;
        let rms = (mean_sq + eps)?.sqrt()?;

        let normalized = x.broadcast_div(&rms)?;
        let weight_broadcast = weight.reshape((1, hidden))?;
        normalized.broadcast_mul(&weight_broadcast)
    } else {
        let hidden = *dims
            .last()
            .ok_or_else(|| candle_core::Error::msg("Empty tensor"))?;
        let weight_2d = weight.reshape((1, hidden))?;

        let variance = x.sqr()?.mean(1)?;
        let x_normed = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&weight_2d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_rms_norm_shape_preserved() {
        let weight = Tensor::ones((32,), DType::F32, &candle_core::Device::Cpu).unwrap();
        let rms = RmsNorm::new(weight, 1e-6);

        let input = Tensor::ones((2, 10, 32), DType::F32, &candle_core::Device::Cpu).unwrap();
        let output = rms.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 32]);
    }

    #[test]
    fn test_rms_norm_2d() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::new(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device).unwrap();
        let weight = Tensor::new(&[1.0, 1.0, 1.0], &device).unwrap();

        let result = rms_norm(&x, &weight, 1e-5).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_rms_norm_3d() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2], &device).unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0], &device).unwrap();

        let result = rms_norm(&x, &weight, 1e-5).unwrap();
        assert_eq!(result.dims(), &[1, 2, 2]);
    }

    #[test]
    fn test_rms_norm_weight_application() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(128, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::ones((2, 128), DType::F32, &device).unwrap();
        let output = norm.forward(&x).unwrap();

        let count = (output.dim(1).unwrap() * output.dim(2).unwrap_or(1)) as f32;
        let sum_out: f32 = output.sum_all().unwrap().to_scalar().unwrap();
        let sum_in: f32 = x.sum_all().unwrap().to_scalar().unwrap();
        let mean_out = sum_out / count;
        let mean_in = sum_in / count;
        assert!((mean_out - mean_in).abs() < 0.5);
    }

    #[test]
    fn test_rms_norm_variance_calculation() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(64, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::full(2.0, (1, 64), &device).unwrap().to_dtype(DType::F32).unwrap();
        let output = norm.forward(&x).unwrap();

        let data = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let first = data[0];
        let all_same = data.iter().all(|v| (v - first).abs() < 1e-6);
        assert!(all_same);
    }

    #[test]
    fn test_rms_norm_forward() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(32, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::randn(0.0, 1.0, (4, 32), &device).unwrap().to_dtype(DType::F32).unwrap();
        let output = norm.forward(&x).unwrap();

        assert_eq!(output.dims(), x.dims());
    }

    #[test]
    fn test_rms_norm_minimal_dim() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(8, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::ones((1, 8), DType::F32, &device).unwrap();
        let output = norm.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 8]);
    }

    #[test]
    fn test_rms_norm_large_dim() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(8192, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::ones((1, 8192), DType::F32, &device).unwrap();
        let output = norm.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 8192]);
    }

    #[test]
    fn test_rms_norm_large_eps() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(64, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 0.1);

        let x = Tensor::randn(0.0, 1.0, (2, 64), &device).unwrap().to_dtype(DType::F32).unwrap();
        let output = norm.forward(&x).unwrap();
        let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_output_finite() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(128, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::randn(-10.0, 10.0, (4, 128), &device).unwrap().to_dtype(DType::F32).unwrap();
        let output = norm.forward(&x).unwrap();
        let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rms_norm_output_scale() {
        let device = candle_core::Device::Cpu;
        let weight = Tensor::ones(64, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let x = Tensor::randn(0.0, 1.0, (2, 64), &device).unwrap().to_dtype(DType::F32).unwrap();
        let output = norm.forward(&x).unwrap();

        let count = output.dim(1).unwrap() * output.dim(2).unwrap_or(1);
        let sum_sq: f32 = output.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        let rms = (sum_sq / count as f32).sqrt();
        assert!(rms < 2.0 && rms > 0.1, "RMS should be in reasonable range");
    }

    #[test]
    fn test_rms_norm_eps_stability() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::ones((1, 64), DType::F32, &device).unwrap();

        let weight1 = Tensor::ones(64, DType::F32, &device).unwrap();
        let weight2 = Tensor::ones(64, DType::F32, &device).unwrap();

        let norm1 = RmsNorm::new(weight1, 1e-8);
        let norm2 = RmsNorm::new(weight2, 1e-2);

        let out1 = norm1.forward(&x).unwrap();
        let out2 = norm2.forward(&x).unwrap();

        assert!(out1.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
        assert!(out2.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite()));
    }
}
