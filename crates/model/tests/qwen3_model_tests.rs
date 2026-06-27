use candle_core::{Device, Tensor};
use vllm_model::qwen3::Qwen3Model;
use vllm_model::qwen3_config::Qwen3Config;
use vllm_traits::ModelBackend;

#[test]
fn test_qwen3_model_forward_cpu() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    // Test forward with single token
    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

    let output = model
        .forward(
            &[1],
            &[vec![42]],
            &[vec![0]],
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.next_tokens.len(), 1);
    assert!(output.next_tokens[0] < 1000);
}

#[test]
fn test_qwen3_model_forward_qk_norm() {
    // Qwen3-0.6B uses q_norm/k_norm
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        has_qk_norm: Some(true),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

    let output = model
        .forward(
            &[1],
            &[vec![42]],
            &[vec![0]],
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.next_tokens.len(), 1);
}

#[test]
fn test_qwen3_model_custom_head_dim() {
    // Qwen3-0.6B: hidden=1024, heads=16, head_dim=128 (not 1024/16=64)
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(1024),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(16),
        num_key_value_heads: Some(8),
        intermediate_size: Some(3072),
        head_dim: Some(128),
        has_qk_norm: Some(true),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

    let output = model
        .forward(
            &[1],
            &[vec![42]],
            &[vec![0]],
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.next_tokens.len(), 1);
}

#[test]
fn test_qwen3_model_batch_forward() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    // Test with batch size 3
    let seq_ids = vec![1u64, 2, 3];
    let input_tokens = vec![vec![1], vec![2], vec![3]];
    let positions = vec![vec![0], vec![0], vec![0]];
    let kv_block_ids = vec![vec![0usize], vec![0], vec![0]];
    let num_computed_tokens = vec![0usize, 0, 0];
    let is_prefill = vec![true, true, true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();

    assert_eq!(output.seq_ids.len(), 3);
    assert_eq!(output.next_tokens.len(), 3);
}

#[test]
fn test_embed_single_text() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(1),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let input_tokens = vec![vec![1u32, 2, 3, 4, 5]];
    let positions = vec![vec![0, 1, 2, 3, 4]];

    let embeddings = model.embed(&input_tokens, &positions).unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 128);
}

#[test]
fn test_embed_batch() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(1),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let input_tokens = vec![vec![1u32, 2, 3], vec![4u32, 5, 6, 7, 8], vec![9u32, 10]];
    let positions = vec![vec![0, 1, 2], vec![0, 1, 2, 3, 4], vec![0, 1]];

    let embeddings = model.embed(&input_tokens, &positions).unwrap();

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.len(), 128);
    }
}

#[test]
fn test_embed_empty_tokens() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(128),
        num_hidden_layers: Some(1),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let input_tokens = vec![vec![]];
    let positions = vec![vec![]];

    let embeddings = model.embed(&input_tokens, &positions).unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 128);
}

#[test]
fn test_qwen3_decode_mode_with_gqa() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(1024),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(16),
        num_key_value_heads: Some(8),
        intermediate_size: Some(3072),
        head_dim: Some(128),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![42]];
    let positions = vec![vec![7]];
    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![7usize];
    let is_prefill = vec![false];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    assert_eq!(output.next_tokens.len(), 1);
}

#[test]
fn test_qwen3_decode_then_continue() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(256),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(512),
        head_dim: Some(64),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let block_ids = vec![vec![0usize]];
    let num_computed = vec![0usize];
    let is_prefill = vec![false];
    let positions = vec![vec![0]];
    let tokens = vec![vec![42u32]];

    let output = model
        .forward(
            &[1],
            &tokens,
            &positions,
            &block_ids,
            &num_computed,
            &is_prefill,
        )
        .unwrap();

    assert_eq!(output.next_tokens.len(), 1);
}

#[test]
fn test_qwen3_kv_cache_gqa_dimensions() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(512),
        num_hidden_layers: Some(1),
        num_attention_heads: Some(8),
        num_key_value_heads: Some(2),
        intermediate_size: Some(1024),
        head_dim: Some(64),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 8).unwrap();

    let seq_ids = vec![1u64, 2u64];
    let input_tokens = vec![vec![1u32, 2, 3], vec![4u32, 5]];
    let positions = vec![vec![0, 1, 2], vec![0, 1]];
    let kv_block_ids = vec![vec![0, 1], vec![2, 3]];
    let num_computed_tokens = vec![0usize, 0];
    let is_prefill = vec![true, true];

    let output = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();

    assert_eq!(output.seq_ids.len(), 2);
    assert_eq!(output.next_tokens.len(), 2);
}

#[test]
fn test_qwen3_decode_logits_2d_shape() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(256),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(512),
        head_dim: Some(64),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let _seq_ids = [1u64];
    let _input_tokens = [vec![42u32]];
    let _positions = [vec![7]];
    let _kv_block_ids = [vec![0usize]];
    let _num_computed_tokens = [7usize];
    let _is_prefill = [false];

    let (logits, _hidden_token) = model
        .forward_with_cache(&[42], 7, &[0], &[7], false)
        .unwrap();

    assert_eq!(
        logits.dims(),
        &[1, 1, 1000],
        "Decode logits should be [1, 1, vocab_size]"
    );
}

#[test]
fn test_qwen3_prefill_logits_3d_shape() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(256),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(512),
        head_dim: Some(64),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 1024).unwrap();

    let (logits, _hidden_token) = model
        .forward_with_cache(
            &[1, 2, 3, 4, 5],
            0,
            &[0, 1, 2, 3, 4],
            &[0, 1, 2, 3, 4],
            true,
        )
        .unwrap();

    assert_eq!(
        logits.dims(),
        &[1, 5, 1000],
        "Prefill logits should be [1, seq_len, vocab_size]"
    );
}

#[test]
fn test_qwen3_decode_sequential_generation() {
    let config = Qwen3Config {
        vocab_size: Some(500),
        hidden_size: Some(256),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(512),
        head_dim: Some(64),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 32).unwrap();

    let block_size = 8;

    for step in 0..20 {
        let block_id = step / block_size;
        let num_computed = step;
        let position = step;

        let (logits, _hidden_token) = model
            .forward_with_cache(&[42], num_computed, &[block_id], &[position], false)
            .unwrap();

        assert_eq!(
            logits.dims(),
            &[1, 1, 500],
            "Step {}: logits shape mismatch",
            step
        );

        let next_token: u32 = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_vec0::<u32>()
            .unwrap();
        assert!(
            next_token < 500,
            "Step {}: next token {} out of range",
            step,
            next_token
        );
    }
}

#[test]
fn test_qwen3_decode_with_gqa_expanded_kv_cache() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(1024),
        num_hidden_layers: Some(3),
        num_attention_heads: Some(16),
        num_key_value_heads: Some(8),
        intermediate_size: Some(3072),
        head_dim: Some(128),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 16).unwrap();

    let output = model
        .forward(
            &[1],
            &[vec![42u32]],
            &[vec![5]],
            &[vec![0usize]],
            &[5],
            &[false],
        )
        .unwrap();

    assert_eq!(output.next_tokens.len(), 1);
    assert!(output.next_tokens[0] < 1000);
}

#[test]
fn test_qwen3_mixed_prefill_and_decode() {
    let config = Qwen3Config {
        vocab_size: Some(500),
        hidden_size: Some(128),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        head_dim: Some(32),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 16).unwrap();

    let _ = model
        .forward(
            &[1],
            &[vec![1u32, 2, 3, 4, 5]],
            &[vec![0, 1, 2, 3, 4]],
            &[vec![0, 1, 2, 3, 4]],
            &[0, 1, 2, 3, 4],
            &[true],
        )
        .unwrap();

    for step in 5..10 {
        let block_id = step / 8;
        let output = model
            .forward(
                &[1],
                &[vec![42u32]],
                &[vec![step]],
                &[vec![block_id]],
                &[step],
                &[false],
            )
            .unwrap();

        assert_eq!(output.next_tokens.len(), 1);
        assert!(output.next_tokens[0] < 500);
    }
}

#[test]
fn test_forward_logits_decode_mode() {
    let config = Qwen3Config {
        vocab_size: Some(300),
        hidden_size: Some(128),
        num_hidden_layers: Some(1),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(2),
        intermediate_size: Some(256),
        head_dim: Some(32),
        ..Default::default()
    };

    let device = Device::Cpu;
    let mut model = Qwen3Model::new(config, device, 8).unwrap();

    let result = model
        .forward_logits(
            &[1],
            &[vec![42u32]],
            &[vec![5]],
            &[vec![0usize]],
            &[5],
            &[false],
        )
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 300);
}
