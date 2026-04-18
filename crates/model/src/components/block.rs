#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::GqaAttention;
use super::LnLayerNorm;
use super::SwiGLU;
use super::AttentionConfig;

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

        let input_ln_weight = Tensor::ones(
            config.hidden_size,
            candle_core::DType::F32,
            vb.device(),
        )?;
        let input_ln_bias = Tensor::zeros(
            config.hidden_size,
            candle_core::DType::F32,
            vb.device(),
        )?;
        let input_layernorm = LnLayerNorm::new(input_ln_weight, input_ln_bias, config.eps);

        let post_ln_weight = Tensor::ones(
            config.hidden_size,
            candle_core::DType::F32,
            vb.device(),
        )?;
        let post_ln_bias = Tensor::zeros(
            config.hidden_size,
            candle_core::DType::F32,
            vb.device(),
        )?;
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
        let x = Tensor::ones((2, 10, 256), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let output = block.forward(&x).unwrap();

        assert_eq!(output.dims(), &[2, 10, 256]);
    }

    #[test]
    fn test_standard_block_single_token() {
        let config = BlockConfig::default();
        let block = StandardBlock::new(&config, None).unwrap();

        let x = Tensor::ones((1, 1, config.hidden_size), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
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
        let x = Tensor::ones((1, 5, 128), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
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

        let x = Tensor::ones((1, 1, 256), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        let output = StandardBlock::forward(&block, &x).unwrap();

        assert_eq!(output.dims(), &[1, 1, 256]);
    }
}
