#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::AttentionConfig;
use super::GqaAttention;
use super::LnLayerNorm;
use super::SwiGLU;

#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub has_qk_norm: bool,
}

impl Default for BlockConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            intermediate_size: 11008,
            head_dim: 128,
            eps: 1e-6,
            has_qk_norm: false,
        }
    }
}

pub struct StandardBlock {
    input_layernorm: LnLayerNorm,
    post_attention_layernorm: LnLayerNorm,
    attention: GqaAttention,
    mlp: SwiGLU,
}

impl StandardBlock {
    pub fn new(config: &BlockConfig, vb: Option<VarBuilder>) -> Result<Self> {
        let vb = vb.unwrap_or_else(|| {
            candle_nn::VarBuilder::zeros(candle_core::DType::F32, &candle_core::Device::Cpu)
        });

        let input_ln_weight =
            Tensor::ones(config.hidden_size, candle_core::DType::F32, vb.device())?;
        let input_ln_bias =
            Tensor::zeros(config.hidden_size, candle_core::DType::F32, vb.device())?;
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, config.eps);

        let post_ln_weight =
            Tensor::ones(config.hidden_size, candle_core::DType::F32, vb.device())?;
        let post_ln_bias = Tensor::zeros(config.hidden_size, candle_core::DType::F32, vb.device())?;
        let post_attention_layernorm = LnLayerNorm::new(post_ln_weight, post_ln_bias, config.eps);

        let attention = GqaAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            Some(vb.clone()),
            AttentionConfig::default(),
            config.has_qk_norm,
        )?;

        let vb_mlp = vb.pp("mlp");
        let mlp = SwiGLU::new(config.hidden_size, config.intermediate_size, Some(vb_mlp))?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        })
    }

    pub fn from_components(
        input_layernorm: LnLayerNorm,
        post_attention_layernorm: LnLayerNorm,
        attention: GqaAttention,
        mlp: SwiGLU,
    ) -> Self {
        Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.attention.forward(&x)?;
        let x = (&x + &residual)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &x + &residual
    }

    pub fn input_layernorm(&self) -> &LnLayerNorm {
        &self.input_layernorm
    }

    pub fn post_attention_layernorm(&self) -> &LnLayerNorm {
        &self.post_attention_layernorm
    }

    pub fn attention(&self) -> &GqaAttention {
        &self.attention
    }

    pub fn mlp(&self) -> &SwiGLU {
        &self.mlp
    }
}

pub trait TransformerBlock: Send + Sync {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        positions: &[usize],
        kv_block_ids: &[usize],
        num_computed: usize,
        is_prefill: bool,
    ) -> Result<Tensor>;

    fn inner_dim(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
}

impl TransformerBlock for StandardBlock {
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        _positions: &[usize],
        _kv_block_ids: &[usize],
        _num_computed: usize,
        _is_prefill: bool,
    ) -> Result<Tensor> {
        StandardBlock::forward(self, hidden_states)
    }

    fn inner_dim(&self) -> usize {
        self.attention().num_heads() * self.attention().head_dim()
    }

    fn num_kv_heads(&self) -> usize {
        self.attention().num_kv_heads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_standard_block_forward_shape() {
        let config = BlockConfig {
            hidden_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 512,
            head_dim: 64,
            eps: 1e-6,
            has_qk_norm: false,
        };

        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::ones(
            (2, 10, 256),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = block.forward(&x).unwrap();

        assert_eq!(output.dims(), &[2, 10, 256]);
    }

    #[test]
    fn test_standard_block_single_token() {
        let config = BlockConfig::default();
        let block = StandardBlock::new(&config, None).unwrap();

        let x = Tensor::ones(
            (1, 1, config.hidden_size),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = block.forward(&x).unwrap();

        assert_eq!(output.dims(), &[1, 1, config.hidden_size]);
    }

    #[test]
    fn test_standard_block_with_qk_norm() {
        let config = BlockConfig {
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 256,
            head_dim: 32,
            eps: 1e-6,
            has_qk_norm: true,
        };

        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::ones(
            (1, 5, 128),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = block.forward(&x).unwrap();

        assert_eq!(output.dims(), &[1, 5, 128]);
    }

    #[test]
    fn test_standard_block_trait_impl() {
        let config = BlockConfig {
            hidden_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 512,
            head_dim: 64,
            eps: 1e-6,
            has_qk_norm: false,
        };

        let block: StandardBlock = StandardBlock::new(&config, None).unwrap();

        assert_eq!(block.inner_dim(), 256);
        assert_eq!(block.num_kv_heads(), 2);

        let x = Tensor::ones(
            (1, 1, 256),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = StandardBlock::forward(&block, &x).unwrap();

        assert_eq!(output.dims(), &[1, 1, 256]);
    }

    #[test]
    fn test_standard_block_deterministic_output() {
        let config = BlockConfig::default();
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, 4, config.hidden_size), &Device::Cpu).unwrap();

        let out1 = block.forward(&x).unwrap();
        let out2 = block.forward(&x).unwrap();

        let diff = (&out1 - &out2).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max_all().unwrap().to_scalar().unwrap();
        assert!(max_diff < 1e-5, "Forward pass should be deterministic");
    }

    #[test]
    fn test_standard_block_residual_connection() {
        let config = BlockConfig {
            hidden_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 256,
            head_dim: 64,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 2, 128), &Device::Cpu).unwrap();

        let output = block.forward(&x).unwrap();
        assert_eq!(output.dims(), x.dims());

        let output_sum: f32 = output.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        let input_sum: f32 = x.abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(
            output_sum > input_sum * 0.1,
            "Residual should preserve signal"
        );
    }

    #[test]
    fn test_standard_block_layernorm_application() {
        let config = BlockConfig {
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 128,
            head_dim: 32,
            eps: 1e-5,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();

        let x = Tensor::ones((1, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();

        let abs_max: f32 = output.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(abs_max.is_finite() && abs_max < 100.0);
    }

    #[test]
    fn test_standard_block_attention_called() {
        let config = BlockConfig {
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 256,
            head_dim: 32,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();

        assert_eq!(block.attention().num_heads(), 4);
        assert_eq!(block.attention().num_kv_heads(), 2);
    }

    #[test]
    fn test_standard_block_mlp_called() {
        let config = BlockConfig {
            hidden_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 512,
            head_dim: 64,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 1, 256), &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();
        let out_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        let out_before_mlp = block.post_attention_layernorm().forward(&x).unwrap();
        let mlp_out = block.mlp().forward(&out_before_mlp).unwrap();

        let mlp_data: Vec<f32> = mlp_out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mlp_data.len(), out_data.len());
    }

    #[test]
    fn test_standard_block_minimal_hidden_size() {
        let config = BlockConfig {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 32,
            head_dim: 8,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::ones((1, 1, 16), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 1, 16]);
    }

    #[test]
    fn test_standard_block_single_head() {
        let config = BlockConfig {
            hidden_size: 128,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 256,
            head_dim: 128,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::ones((1, 1, 128), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 1, 128]);
    }

    #[test]
    fn test_standard_block_large_batch() {
        let config = BlockConfig {
            hidden_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 512,
            head_dim: 64,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::ones((64, 4, 256), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();
        assert_eq!(output.dims(), &[64, 4, 256]);
    }

    #[test]
    fn test_standard_block_extreme_intermediate_size() {
        let config = BlockConfig {
            hidden_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 512,
            head_dim: 64,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::ones((1, 1, 128), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();
        assert!(output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn test_standard_block_output_finite() {
        let config = BlockConfig::default();
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::randn(-1.0f32, 1.0, (2, 4, config.hidden_size), &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();

        let data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "Output must not contain NaN or Inf"
        );
    }

    #[test]
    fn test_standard_block_output_magnitude() {
        let config = BlockConfig::default();
        let block = StandardBlock::new(&config, None).unwrap();
        let x = Tensor::randn(-1.0f32, 1.0, (1, 2, config.hidden_size), &Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();

        let abs_max: f32 = output.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(
            abs_max < 100.0,
            "Output magnitude unreasonable: {}",
            abs_max
        );
    }

    #[test]
    fn test_standard_block_eps_stability() {
        let x = Tensor::randn(0.0f32, 1.0, (1, 1, 64), &Device::Cpu).unwrap();

        let config1 = BlockConfig {
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 128,
            head_dim: 32,
            eps: 1e-6,
            has_qk_norm: false,
        };
        let config2 = BlockConfig {
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 128,
            head_dim: 32,
            eps: 1e-2,
            has_qk_norm: false,
        };

        let block1 = StandardBlock::new(&config1, None).unwrap();
        let block2 = StandardBlock::new(&config2, None).unwrap();

        let out1 = block1.forward(&x).unwrap();
        let out2 = block2.forward(&x).unwrap();

        assert!(out1
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
        assert!(out2
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
