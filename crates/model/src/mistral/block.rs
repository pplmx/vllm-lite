#![allow(clippy::module_name_repetitions)]
//! Mistral decoder block — shared RoPE-GQA factory.

pub use crate::components::decoder_block::{
    RopeGqaDecoderBlock as MistralBlock, block_from_weights, new_block,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use candle_core::Tensor;

    #[test]
    fn test_mistral_block_forward() {
        let config = ModelConfig::test_tiny_for(crate::config::Architecture::Mistral);
        let block = new_block(&config, 0).unwrap();
        let input = Tensor::ones(
            (1, 4, config.hidden_size),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, config.hidden_size]);
    }

    #[test]
    fn test_mistral_block_sliding_window_config() {
        let config = ModelConfig::test_tiny_for(crate::config::Architecture::Mistral);
        let _block = new_block(&config, 0).unwrap();
        assert_eq!(config.sliding_window, Some(4096));
    }
}
