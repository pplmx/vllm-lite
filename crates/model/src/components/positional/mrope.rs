use crate::qwen3_config::Qwen3Config;
use candle_core::{Device, Result as CandleResult, Tensor};

#[derive(Clone)]
#[allow(dead_code)]
pub struct MRoPE {
    pub(crate) dim: usize,
    pub(crate) theta: f32,
    pub(crate) sections: Vec<usize>,
    pub(crate) partial_rotary_factor: f32,
}

impl MRoPE {
    pub fn new(dim: usize, theta: f32, sections: Vec<usize>, partial_rotary_factor: f32) -> Self {
        Self {
            dim,
            theta,
            sections,
            partial_rotary_factor,
        }
    }

    pub fn from_config(config: &Qwen3Config) -> Self {
        let rope_params = config.rope_parameters();

        let theta = config.rope_theta();
        let head_dim = config.head_dim();

        let sections = rope_params
            .and_then(|rp| rp.mrope_section.clone())
            .unwrap_or_else(|| vec![head_dim / 3, head_dim / 3, head_dim - 2 * (head_dim / 3)]);

        let partial_rotary_factor = rope_params
            .and_then(|rp| rp.partial_rotary_factor)
            .unwrap_or(0.25);

        Self {
            dim: head_dim,
            theta,
            sections,
            partial_rotary_factor,
        }
    }

    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[i64],
    ) -> CandleResult<(Tensor, Tensor)> {
        let (_batch, _seq, _heads, _head_dim) = q.dims4()?;

        let mut q_splits = Vec::with_capacity(self.sections.len());
        let mut q_offset = 0;
        for (i, &section) in self.sections.iter().enumerate() {
            let q_section = q.narrow(3, q_offset, section)?;
            q_splits.push(self.apply_rope_section(&q_section, positions, i)?);
            q_offset += section;
        }

        let mut k_splits = Vec::with_capacity(self.sections.len());
        let mut k_offset = 0;
        for (i, &section) in self.sections.iter().enumerate() {
            let k_section = k.narrow(3, k_offset, section)?;
            k_splits.push(self.apply_rope_section(&k_section, positions, i)?);
            k_offset += section;
        }

        let q_out = Tensor::cat(&q_splits, 3)?;
        let k_out = Tensor::cat(&k_splits, 3)?;

        Ok((q_out, k_out))
    }

    fn apply_rope_section(
        &self,
        x: &Tensor,
        positions: &[i64],
        section_idx: usize,
    ) -> CandleResult<Tensor> {
        let (_batch, seq, _heads, dim) = x.dims4()?;
        let half_dim = dim / 2;

        let x_even = x.narrow(3, 0, half_dim)?;
        let x_odd = x.narrow(3, half_dim, half_dim)?;

        let freq = self.compute_freqs(positions, section_idx)?;
        let freq_sin = freq.sin()?;
        let freq_cos = freq.cos()?;

        let freq_cos = freq_cos.reshape((1, seq, 1, half_dim))?;
        let freq_sin = freq_sin.reshape((1, seq, 1, half_dim))?;

        let x_even_rot = x_even.broadcast_mul(&freq_cos)?;
        let x_odd_rot = x_odd.broadcast_mul(&freq_sin)?;
        let rotated = (x_even_rot - x_odd_rot)?;

        Tensor::cat(&[&rotated, &x.narrow(3, half_dim, half_dim)?], 3)
    }

    fn compute_freqs(&self, positions: &[i64], section_idx: usize) -> CandleResult<Tensor> {
        let seq_len = positions.len();
        let device = &Device::Cpu;
        let half_dim = self.sections[section_idx] / 2;
        let freqs = Tensor::from_vec(
            positions
                .iter()
                .flat_map(|&pos| {
                    (0..half_dim).map(move |i| {
                        let freq = self
                            .theta
                            .powf(-2.0 * (i as f32) / (self.sections[section_idx] as f32));
                        freq * (pos as f32)
                    })
                })
                .collect::<Vec<_>>(),
            (seq_len, half_dim),
            device,
        )?;
        Ok(freqs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_config::RopeParameters;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_mrope_creation() {
        let config = Qwen3Config {
            rope_theta: Some(10000000.0),
            head_dim: Some(256),
            rope_parameters: Some(RopeParameters {
                rope_type: Some("default".to_string()),
                rope_theta: Some(10000000.0),
                partial_rotary_factor: Some(0.5),
                mrope_section: Some(vec![85, 85, 86]),
                mrope_interleaved: Some(true),
            }),
            ..Default::default()
        };

        let rope = MRoPE::from_config(&config);
        assert_eq!(rope.theta, 10000000.0);
        assert_eq!(rope.dim, 256);
        assert_eq!(rope.sections, vec![85, 85, 86]);
    }

    #[test]
    fn test_mrope_new() {
        let rope = MRoPE::new(32, 10000.0, vec![10, 10, 12], 0.25);
        assert_eq!(rope.dim, 32);
        assert_eq!(rope.theta, 10000.0);
        assert_eq!(rope.sections, vec![10, 10, 12]);
    }

    #[test]
    fn test_mrope_output_shape() {
        let rope = MRoPE::new(12, 10000.0, vec![4, 4, 4], 0.25);
        let device = Device::Cpu;

        let q = Tensor::ones((2, 4, 8, 12), DType::F32, &device).unwrap();
        let k = Tensor::ones((2, 4, 2, 12), DType::F32, &device).unwrap();
        let positions: Vec<i64> = vec![0, 1, 2, 3];

        let (q_out, k_out) = rope.apply(&q, &k, &positions).unwrap();

        assert_eq!(q_out.dims(), q.dims());
        assert_eq!(k_out.dims(), k.dims());
    }

    #[test]
    fn test_mrope_different_positions_different_output() {
        let rope = MRoPE::new(12, 10000.0, vec![4, 4, 4], 0.25);
        let device = Device::Cpu;

        let q = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();

        let pos_0: Vec<i64> = vec![0, 1];
        let pos_5: Vec<i64> = vec![5, 6];

        let (_, k_0) = rope.apply(&q, &k, &pos_0).unwrap();
        let (_, k_5) = rope.apply(&q, &k, &pos_5).unwrap();

        let diff = (&k_0 - &k_5)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-3,
            "RoPE should produce different outputs for different positions, got diff={}",
            diff
        );
    }

    #[test]
    fn test_mrope_deterministic() {
        let rope = MRoPE::new(12, 10000.0, vec![4, 4, 4], 0.25);
        let device = Device::Cpu;

        let q = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();
        let k = Tensor::ones((1, 2, 2, 12), DType::F32, &device).unwrap();
        let positions: Vec<i64> = vec![3, 4];

        let (q1, k1) = rope.apply(&q, &k, &positions).unwrap();
        let (q2, k2) = rope.apply(&q, &k, &positions).unwrap();

        let q_diff = (&q1 - &q2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let k_diff = (&k1 - &k2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert_eq!(q_diff, 0.0, "RoPE should be deterministic for q");
        assert_eq!(k_diff, 0.0, "RoPE should be deterministic for k");
    }

    #[test]
    fn test_mrope_section_validation() {
        let head_dim = 12;
        let sections = [4, 4, 4];
        let total: usize = sections.iter().sum();

        assert_eq!(total, head_dim, "Sections should sum to head_dim");
        assert!(
            sections.iter().all(|s| s % 2 == 0),
            "Each section should be even for half-dim split"
        );
    }
}
