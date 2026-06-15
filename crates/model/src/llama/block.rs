//! Llama decoder block — shared RoPE-GQA factory.

pub use crate::components::decoder_block::{
    RopeGqaDecoderBlock as LlamaBlock, block_from_weights, new_block,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::{DType, Device, Tensor};

    fn test_config() -> ModelConfig {
        ModelConfig::test_tiny()
    }

    #[test]
    fn test_llama_block_forward_shape() {
        let config = test_config();
        let block = new_block(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, config.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, config.hidden_size]);
    }

    #[test]
    fn test_llama_block_single_token() {
        let config = test_config();
        let block = new_block(&config, 0).unwrap();

        let input = Tensor::ones((1, 1, config.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, config.hidden_size]);
    }

    #[test]
    fn test_llama_block_different_batch_sizes() {
        let config = test_config();

        for batch_size in [1usize, 2, 4] {
            let block = new_block(&config, 0).unwrap();
            let input = Tensor::ones(
                (batch_size, 5, config.hidden_size),
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let output = block.forward(&input).unwrap();
            assert_eq!(output.dims()[0], batch_size);
        }
    }

    #[test]
    #[ignore = "slow integration test - run with --ignored for full model validation"]
    fn test_llama_block_full_size() {
        let config = ModelConfig::llama_7b();
        let block = new_block(&config, 0).unwrap();

        let input = Tensor::ones((2, 10, 4096), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 4096]);
    }
}
