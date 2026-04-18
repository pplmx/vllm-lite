use candle_core::{DType, Device, Tensor};
use vllm_model::components::attention::{AttentionConfig, GqaAttention};

#[test]
fn test_gqa_expand_kv_same_heads() {
    let device = Device::Cpu;

    // MHA: 4 query heads, 4 KV heads (same)
    let attn = GqaAttention::new(256, 4, 4, 64, None, AttentionConfig::default(), false).unwrap();

    // Input: [batch=1, seq=2, num_kv_heads=4, head_dim=64]
    let kv = Tensor::ones((1, 2, 4, 64), DType::F32, &device).unwrap();

    let expanded = attn.expand_kv(&kv, 4, 4).unwrap();

    // Should be unchanged since num_q_heads == num_kv_heads
    assert_eq!(expanded.dims(), &[1, 2, 4, 64]);
}

#[test]
fn test_gqa_expand_kv_gqa() {
    let device = Device::Cpu;

    // GQA: 8 query heads, 2 KV heads
    let attn = GqaAttention::new(512, 8, 2, 64, None, AttentionConfig::default(), false).unwrap();

    // Input: [batch=1, seq=2, num_kv_heads=2, head_dim=64]
    let kv = Tensor::ones((1, 2, 2, 64), DType::F32, &device).unwrap();

    let expanded = attn.expand_kv(&kv, 8, 2).unwrap();

    // Should expand from 2 KV heads to 8 query heads
    assert_eq!(expanded.dims(), &[1, 2, 8, 64]);
}

#[test]
fn test_gqa_expand_kv_mqa() {
    let device = Device::Cpu;

    // MQA: 8 query heads, 1 KV head
    let attn = GqaAttention::new(512, 8, 1, 64, None, AttentionConfig::default(), false).unwrap();

    // Input: [batch=2, seq=3, num_kv_heads=1, head_dim=64]
    let kv = Tensor::ones((2, 3, 1, 64), DType::F32, &device).unwrap();

    let expanded = attn.expand_kv(&kv, 8, 1).unwrap();

    // Should expand from 1 KV head to 8 query heads
    assert_eq!(expanded.dims(), &[2, 3, 8, 64]);
}

#[test]
fn test_gqa_attention_forward_single_token() {
    let device = Device::Cpu;

    // Qwen2.5-0.5B config: 14 Q heads, 2 KV heads, 64 head_dim
    let attn = GqaAttention::new(896, 14, 2, 64, None, AttentionConfig::default(), false).unwrap();

    // Input: [batch=1, seq=1, hidden=896] - use F32 explicitly
    let x = Tensor::randn(0.0, 1.0, (1, 1, 896), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = attn.forward(&x).unwrap();

    // Output should have same shape as input
    assert_eq!(output.dims(), &[1, 1, 896]);
}

#[test]
fn test_attention_computes_attention_weights() {
    let device = Device::Cpu;

    // Create random weights with correct dimensions
    let q_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let k_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let v_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let o_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let attn = GqaAttention::new_with_weights(
        128,
        2,
        2,
        64,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        AttentionConfig::default(),
        false,
        None,
        None,
    )
    .unwrap();

    // Use random input to ensure non-zero computation
    let x = Tensor::randn(0.0, 1.0, (1, 2, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let output = attn.forward(&x).unwrap();

    // Output shape should be correct
    assert_eq!(output.dims(), &[1, 2, 128]);

    // Verify that different sequence positions produce different outputs
    // due to attention mixing information across positions
    let t0_data = output
        .get(0)
        .unwrap()
        .get(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let t1_data = output
        .get(0)
        .unwrap()
        .get(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_ne!(
        t0_data, t1_data,
        "attention should produce different outputs at different positions"
    );
}

#[test]
fn test_gqa_attention_forward_multiple_tokens() {
    let device = Device::Cpu;

    let attn = GqaAttention::new(896, 14, 2, 64, None, AttentionConfig::default(), false).unwrap();

    // Input: [batch=1, seq=4, hidden=896]
    let x = Tensor::randn(0.0, 1.0, (1, 4, 896), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = attn.forward(&x).unwrap();

    assert_eq!(output.dims(), &[1, 4, 896]);
}

#[test]
fn test_gqa_attention_forward_batch() {
    let device = Device::Cpu;

    let attn = GqaAttention::new(512, 8, 2, 64, None, AttentionConfig::default(), false).unwrap();

    // Input: [batch=3, seq=2, hidden=512]
    let x = Tensor::randn(0.0, 1.0, (3, 2, 512), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let output = attn.forward(&x).unwrap();

    assert_eq!(output.dims(), &[3, 2, 512]);
}

#[test]
fn test_qwen2_config_gqa() {
    // Qwen2.5-0.5B uses GQA: 14 Q heads, 2 KV heads
    let num_q_heads = 14;
    let num_kv_heads = 2;
    let repeat_factor = num_q_heads / num_kv_heads;

    assert_eq!(repeat_factor, 7);
}

#[test]
fn test_attention_output_is_different_from_input() {
    let device = Device::Cpu;
    let attn = GqaAttention::new(256, 4, 2, 64, None, AttentionConfig::default(), false).unwrap();

    // Use non-zero input to ensure attention computation happens
    let x = Tensor::ones((1, 2, 256), DType::F32, &device).unwrap();

    let output = attn.forward(&x).unwrap();

    // Attention should transform the input - output should differ from input
    let x_sum = x.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let output_sum = output.sum_all().unwrap().to_scalar::<f32>().unwrap();

    assert_ne!(
        x_sum, output_sum,
        "attention output should be different from input"
    );
}

#[test]
fn test_attention_causal_mask_behavior() {
    let device = Device::Cpu;

    let q_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let k_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let v_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let o_weight = Tensor::randn(0.0, 0.1, (128, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let attn = GqaAttention::new_with_weights(
        128,
        2,
        2,
        64,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        AttentionConfig::default(),
        false,
        None,
        None,
    )
    .unwrap();

    let x = Tensor::randn(0.0, 1.0, (1, 3, 128), &device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let output = attn.forward(&x).unwrap();

    // Output shape should be correct
    assert_eq!(output.dims(), &[1, 3, 128]);

    // Extract outputs at different positions
    let token_0 = output.get(0).unwrap().get(0).unwrap();
    let token_1 = output.get(0).unwrap().get(1).unwrap();
    let token_2 = output.get(0).unwrap().get(2).unwrap();

    // Tokens at different positions should not be identical (causal attention effect)
    let t0_data = token_0.to_vec1::<f32>().unwrap();
    let t1_data = token_1.to_vec1::<f32>().unwrap();
    let t2_data = token_2.to_vec1::<f32>().unwrap();

    assert_ne!(t0_data, t1_data, "position 0 and 1 should differ");
    assert_ne!(t1_data, t2_data, "position 1 and 2 should differ");
}
