//! `RoPE` unit tests (pure tensor math, no checkpoint).

#[test]
fn test_qwen3_rope_position_encoding() {
    use candle_core::{Device, Tensor};
    use vllm_model::components::positional::apply_rope;

    let device = Device::Cpu;
    let head_dim = 128;
    let theta = 1_000_000.0f32;

    let batch = 1;
    let seq_len = 3;
    let num_heads = 2;
    let query = Tensor::randn(0.0, 1.0, (batch, seq_len, num_heads, head_dim), &device)
        .expect("Failed to create query tensor")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert to F32");

    let positions: Vec<i64> = vec![0, 5, 10];
    let rotated = apply_rope(&query, &positions, theta).expect("RoPE failed");

    assert_eq!(rotated.dims(), query.dims());

    let rotated_at_5 = rotated.narrow(1, 1, 1).unwrap();
    let rot_data = rotated_at_5
        .reshape((num_heads * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let norm: f32 = rot_data.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(norm > 0.01, "RoPE output should have non-trivial magnitude");

    let rotated_at_10 = rotated.narrow(1, 2, 1).unwrap();
    let rot_data_5 = rotated_at_5
        .reshape((num_heads * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let rot_data_10 = rotated_at_10
        .reshape((num_heads * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // invariant: num_heads * head_dim is bounded by the small test tensor
    // dimensions; f32 precision loss is acceptable for the test mean diff.
    #[allow(clippy::cast_precision_loss)]
    let diff_pos: f32 = rot_data_5
        .iter()
        .zip(rot_data_10.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / (num_heads * head_dim) as f32;

    assert!(
        diff_pos > 0.001,
        "Different positions should produce different results"
    );
}

#[test]
fn test_qwen3_rope_consistency_and_norm() {
    use candle_core::{Device, Tensor};
    use vllm_model::components::positional::apply_rope;

    let device = Device::Cpu;
    let head_dim = 128;
    let theta = 1_000_000.0f32;

    let query = Tensor::randn(0.0, 1.0, (1, 1, 2, head_dim), &device)
        .expect("Failed to create query tensor")
        .to_dtype(candle_core::DType::F32)
        .expect("Failed to convert to F32");
    let positions = vec![5i64];

    let rotated1 = apply_rope(&query, &positions, theta).expect("RoPE failed");
    let rotated2 = apply_rope(&query, &positions, theta).expect("RoPE failed");

    let rot_data1 = rotated1
        .reshape((2 * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let rot_data2 = rotated2
        .reshape((2 * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // invariant: 2 * head_dim is bounded by the small test tensor dimensions;
    // f32 precision loss is acceptable for the determinism check.
    #[allow(clippy::cast_precision_loss)]
    let diff: f32 = rot_data1
        .iter()
        .zip(rot_data2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / (2 * head_dim) as f32;

    assert!(diff < 1e-6, "RoPE should be deterministic");

    let orig_data = query
        .reshape((2 * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let rotated_data = rotated1
        .reshape((2 * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let orig_norm: f32 = orig_data.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let rotated_norm: f32 = rotated_data.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_diff = (orig_norm - rotated_norm).abs() / orig_norm;

    assert!(
        norm_diff < 0.01,
        "RoPE should approximately preserve L2 norm"
    );
}
