use candle_core::{Device, Tensor};
use vllm_model::qwen3::attention::{AttentionConfig, GqaAttention};

const THETA: f32 = 10000.0;

#[test]
fn test_tiled_attention_short_seq() {
    let device = Device::Cpu;
    let config = AttentionConfig {
        tile_size: Some(16),
        use_fused: true,
    };
    let _attn = GqaAttention::new(256, 4, 4, 64, THETA, None, config, false).unwrap();

    let _x = Tensor::randn(0.0, 1.0, (1, 8, 256), &device).unwrap();
}

#[test]
fn test_tiled_attention_long_seq() {
    let device = Device::Cpu;
    let config = AttentionConfig {
        tile_size: Some(16),
        use_fused: true,
    };
    let _attn = GqaAttention::new(256, 4, 4, 64, THETA, None, config, false).unwrap();

    let _x = Tensor::randn(0.0, 1.0, (1, 32, 256), &device).unwrap();
}

#[test]
fn test_tiled_attention_no_tile() {
    let device = Device::Cpu;
    let config = AttentionConfig {
        tile_size: None,
        use_fused: true,
    };
    let _attn = GqaAttention::new(256, 4, 4, 64, THETA, None, config, false).unwrap();

    let _x = Tensor::randn(0.0, 1.0, (1, 64, 256), &device).unwrap();
}
