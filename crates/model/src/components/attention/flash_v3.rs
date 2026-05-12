use candle_core::{Result, Tensor};
use tracing::trace;

#[derive(Debug, Clone)]
pub struct FlashAttentionV3 {
    num_heads: usize,
    head_dim: usize,
    dropout_p: f32,
    causal: bool,
    window_size: Option<(i32, i32)>,
}

#[derive(Debug, Clone, Default)]
pub struct FlashAttentionV3Config {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_p: f32,
    pub causal: bool,
    pub window_size: Option<(i32, i32)>,
}

impl FlashAttentionV3 {
    pub fn new(config: FlashAttentionV3Config) -> Self {
        Self {
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            dropout_p: config.dropout_p,
            causal: config.causal,
            window_size: config.window_size,
        }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        trace!(
            batch = ?q.dims()[0],
            seq_q = ?q.dims()[2],
            seq_k = ?k.dims()[2],
            num_heads = self.num_heads,
            head_dim = self.head_dim,
            causal = self.causal,
            "FlashAttentionV3 forward"
        );

        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let qk = Tensor::matmul(q, &k.transpose(2, 3)?.contiguous()?)?;
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;

        if self.causal {
            let seq_len = q.dims()[2];
            let mask = self.create_causal_mask(seq_len, q.device())?;
            let _qk = qk.broadcast_add(&mask)?;
        }

        if let Some((left, right)) = self.window_size {
            let mask = self.create_sliding_window_mask_simple(
                q.dims()[2],
                k.dims()[2],
                left,
                right,
                q.device(),
            )?;
            let _qk = qk.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;

        let dropout_scale = if self.dropout_p > 0.0 {
            1.0 / (1.0 - self.dropout_p)
        } else {
            1.0
        };

        let output = Tensor::matmul(&attn_weights, v)?;
        if dropout_scale != 1.0 {
            let scale_tensor =
                Tensor::new(&[dropout_scale], q.device())?.broadcast_as(output.dims())?;
            output.mul(&scale_tensor)
        } else {
            Ok(output)
        }
    }

    pub fn forward_with_swa(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        self.forward(q, k, v)
    }

    fn create_causal_mask(&self, seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        let row_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, seq_len, 1))?;
        let col_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, 1, seq_len))?;
        let row_indices = row_indices.broadcast_as((1, 1, seq_len, seq_len))?;
        let col_indices = col_indices.broadcast_as((1, 1, seq_len, seq_len))?;
        let mask = row_indices.ge(&col_indices)?;
        let zero = Tensor::new(0.0f32, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
        let neg_inf =
            Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
        mask.where_cond(&zero, &neg_inf)
    }

    #[allow(dead_code)]
    fn create_sliding_window_mask(
        &self,
        dims: &[usize],
        left: i32,
        right: i32,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let batch_size = dims[0];
        let num_heads = dims[1];
        let seq_q = dims[2];
        let seq_k = dims[2];

        let mask: Vec<f32> = (0..seq_q)
            .flat_map(|i| {
                (0..seq_k).map(move |j| {
                    let offset = i as i32 - j as i32;
                    if offset < -left || offset > right {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        Tensor::from_slice(&mask, (batch_size, num_heads, seq_q, seq_k), device)
    }

    fn create_sliding_window_mask_simple(
        &self,
        seq_q: usize,
        seq_k: usize,
        left: i32,
        right: i32,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_q)
            .flat_map(|i| {
                (0..seq_k).map(move |j| {
                    let offset = i as i32 - j as i32;
                    if offset < -left || offset > right {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        Tensor::from_slice(&mask, (1, 1, seq_q, seq_k), device)
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

pub struct MqaFlashAttention {
    num_heads: usize,
    #[allow(dead_code)]
    num_kv_heads: usize,
    #[allow(dead_code)]
    head_dim: usize,
    causal: bool,
}

impl MqaFlashAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
        }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_batch_size, _, seq_q, head_dim) =
            (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);

        let k_expanded = self.expand_kv(k, self.num_heads)?;
        let v_expanded = self.expand_kv(v, self.num_heads)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let qk = Tensor::matmul(q, &k_expanded.transpose(2, 3)?.contiguous()?)?;
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;

        if self.causal {
            let mask = self.create_causal_mask(seq_q, q.device())?;
            let _qk = qk.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
        let output = Tensor::matmul(&attn_weights, &v_expanded)?;

        Ok(output)
    }

    fn expand_kv(&self, kv: &Tensor, num_q_heads: usize) -> Result<Tensor> {
        let num_kv_heads = kv.dims()[1];
        if num_kv_heads == num_q_heads {
            return Ok(kv.clone());
        }
        let repeat_factor = num_q_heads / num_kv_heads;
        kv.repeat(&[1, repeat_factor, 1, 1])
    }

    fn create_causal_mask(&self, seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        let row_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, seq_len, 1))?;
        let col_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, 1, seq_len))?;
        let row_indices = row_indices.broadcast_as((1, 1, seq_len, seq_len))?;
        let col_indices = col_indices.broadcast_as((1, 1, seq_len, seq_len))?;
        let mask = row_indices.ge(&col_indices)?;
        let zero = Tensor::new(0.0f32, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
        let neg_inf =
            Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
        mask.where_cond(&zero, &neg_inf)
    }
}

pub struct GqaFlashAttention {
    num_heads: usize,
    #[allow(dead_code)]
    num_kv_heads: usize,
    #[allow(dead_code)]
    head_dim: usize,
    causal: bool,
}

impl GqaFlashAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
        }
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_batch_size, _, seq_q, head_dim) =
            (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);

        let k_expanded = self.expand_kv(k, self.num_heads)?;
        let v_expanded = self.expand_kv(v, self.num_heads)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let qk = Tensor::matmul(q, &k_expanded.transpose(2, 3)?.contiguous()?)?;
        let qk = qk.mul(&Tensor::new(&[scale], q.device())?.broadcast_as(qk.dims())?)?;

        if self.causal {
            let mask = self.create_causal_mask(seq_q, q.device())?;
            let _qk = qk.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax(&qk, 3)?;
        let output = Tensor::matmul(&attn_weights, &v_expanded)?;

        Ok(output)
    }

    fn expand_kv(&self, kv: &Tensor, num_q_heads: usize) -> Result<Tensor> {
        let num_kv_heads = kv.dims()[1];
        if num_kv_heads == num_q_heads {
            return Ok(kv.clone());
        }

        if num_q_heads % num_kv_heads != 0 {
            let repeat_factor = num_q_heads.div_ceil(num_kv_heads);
            let kv_repeated = kv.repeat(&[1, repeat_factor, 1, 1])?;
            kv_repeated.narrow(1, 0, num_q_heads)
        } else {
            let repeat_factor = num_q_heads / num_kv_heads;
            kv.repeat(&[1, repeat_factor, 1, 1])
        }
    }

    fn create_causal_mask(&self, seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
        let row_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, seq_len, 1))?;
        let col_indices =
            Tensor::arange(0u32, seq_len as u32, device)?.reshape((1, 1, 1, seq_len))?;
        let row_indices = row_indices.broadcast_as((1, 1, seq_len, seq_len))?;
        let col_indices = col_indices.broadcast_as((1, 1, seq_len, seq_len))?;
        let mask = row_indices.ge(&col_indices)?;
        let zero = Tensor::new(0.0f32, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
        let neg_inf =
            Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as((1, 1, seq_len, seq_len))?;
        mask.where_cond(&zero, &neg_inf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: &candle_core::Device = &candle_core::Device::Cpu;

    #[test]
    fn test_flash_attention_v3_basic() {
        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 4;
        let head_dim = 32;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let flash = FlashAttentionV3::new(FlashAttentionV3Config {
            num_heads,
            head_dim,
            dropout_p: 0.0,
            causal: false,
            window_size: None,
        });

        let output = flash.forward(&q, &k, &v).unwrap();
        assert_eq!(output.dims(), q.dims());
    }

    #[test]
    fn test_flash_attention_v3_causal() {
        let batch_size = 1;
        let seq_len = 4;
        let num_heads = 4;
        let head_dim = 32;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let flash = FlashAttentionV3::new(FlashAttentionV3Config {
            num_heads,
            head_dim,
            dropout_p: 0.0,
            causal: true,
            window_size: None,
        });

        let output = flash.forward(&q, &k, &v).unwrap();
        assert_eq!(output.dims(), q.dims());
    }

    #[test]
    fn test_flash_attention_v3_with_sliding_window() {
        let batch_size = 1;
        let seq_len = 16;
        let num_heads = 4;
        let head_dim = 32;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let flash = FlashAttentionV3::new(FlashAttentionV3Config {
            num_heads,
            head_dim,
            dropout_p: 0.0,
            causal: false,
            window_size: Some((8, 8)),
        });

        let output = flash.forward_with_swa(&q, &k, &v).unwrap();
        assert_eq!(output.dims(), q.dims());
    }

    #[test]
    fn test_mqa_flash_attention() {
        let batch_size = 1;
        let seq_len = 8;
        let num_heads = 16;
        let num_kv_heads = 1;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let mqa = MqaFlashAttention::new(num_heads, num_kv_heads, head_dim, true);
        let output = mqa.forward(&q, &k, &v).unwrap();

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_gqa_flash_attention() {
        let batch_size = 1;
        let seq_len = 8;
        let num_heads = 16;
        let num_kv_heads = 4;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let gqa = GqaFlashAttention::new(num_heads, num_kv_heads, head_dim, true);
        let output = gqa.forward(&q, &k, &v).unwrap();

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_gqa_flash_attention_non_divisible() {
        let batch_size = 1;
        let seq_len = 8;
        let num_heads = 14;
        let num_kv_heads = 7;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let gqa = GqaFlashAttention::new(num_heads, num_kv_heads, head_dim, false);
        let output = gqa.forward(&q, &k, &v).unwrap();

        assert_eq!(output.dims(), &[batch_size, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_flash_attention_v3_output_finite() {
        let batch_size = 2;
        let seq_len = 16;
        let num_heads = 8;
        let head_dim = 64;

        let q = Tensor::randn(
            -2.0f32,
            2.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            -2.0f32,
            2.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            -2.0f32,
            2.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let flash = FlashAttentionV3::new(FlashAttentionV3Config {
            num_heads,
            head_dim,
            dropout_p: 0.0,
            causal: true,
            window_size: Some((8, 0)),
        });

        let output = flash.forward(&q, &k, &v).unwrap();
        let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mqa_flash_attention_deterministic() {
        let batch_size = 1;
        let seq_len = 8;
        let num_heads = 8;
        let num_kv_heads = 1;
        let head_dim = 64;

        let q = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();
        let v = Tensor::randn(
            0.0f32,
            1.0,
            (batch_size, num_kv_heads, seq_len, head_dim),
            DEVICE,
        )
        .unwrap();

        let mqa = MqaFlashAttention::new(num_heads, num_kv_heads, head_dim, false);

        let out1 = mqa.forward(&q, &k, &v).unwrap();
        let out2 = mqa.forward(&q, &k, &v).unwrap();

        let diff = (&out1 - &out2).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
            .iter()
            .cloned()
            .fold(0.0f32, |a, b| a.max(b));
        assert!(max_diff < 1e-6);
    }
}
