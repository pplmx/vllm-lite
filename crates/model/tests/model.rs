use vllm_model::qwen3_config::Qwen3Config;
use vllm_testing::FakeModel;
use vllm_traits::ModelBackend;

#[test]
fn test_fake_model_output_count() {
    let mut model = FakeModel::new(1000);
    let seq_ids = vec![1u64, 2, 3];
    let input_tokens = vec![vec![1u32, 2], vec![3, 4], vec![5, 6]];
    let positions = vec![vec![0usize, 1], vec![0, 1], vec![0, 1]];
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
fn test_fake_model_deterministic() {
    let mut model = FakeModel::new(42);
    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![1u32, 2, 3]];
    let positions = vec![vec![0usize, 1, 2]];
    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

    let output1 = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();
    let output2 = model
        .forward(
            &seq_ids,
            &input_tokens,
            &positions,
            &kv_block_ids,
            &num_computed_tokens,
            &is_prefill,
        )
        .unwrap();

    // Same input should produce same output
    assert_eq!(output1.next_tokens, output2.next_tokens);
}

#[test]
fn test_fake_model_different_seqs_different_output() {
    let mut model = FakeModel::new(100);
    let seq_ids = vec![1u64, 2u64];
    let input_tokens = vec![vec![1u32], vec![1u32]];
    let positions = vec![vec![0usize], vec![0usize]];
    let kv_block_ids = vec![vec![0usize], vec![0]];
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

    // Different sequence IDs should produce different tokens
    assert_ne!(
        output.next_tokens[0], output.next_tokens[1],
        "different seq_ids should produce different tokens"
    );
}

#[test]
fn test_fake_model_batch_size_respected() {
    let mut model = FakeModel::new(1000);

    // Test various batch sizes
    for batch_size in [1, 2, 5, 10] {
        let seq_ids: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
        let input_tokens: Vec<Vec<u32>> = (0..batch_size).map(|_| vec![1]).collect();
        let positions: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
        let kv_block_ids: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
        let num_computed_tokens: Vec<usize> = (0..batch_size).map(|_| 0).collect();
        let is_prefill: Vec<bool> = (0..batch_size).map(|_| true).collect();

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
        assert_eq!(
            output.seq_ids.len(),
            batch_size,
            "batch size {} not respected",
            batch_size
        );
        assert_eq!(
            output.next_tokens.len(),
            batch_size,
            "batch size {} not respected",
            batch_size
        );
    }
}

#[test]
fn test_model_empty_batch() {
    let mut model = FakeModel::new(1000);
    let seq_ids: Vec<u64> = vec![];
    let input_tokens: Vec<Vec<u32>> = vec![];
    let positions: Vec<Vec<usize>> = vec![];
    let kv_block_ids: Vec<Vec<usize>> = vec![];
    let num_computed_tokens: Vec<usize> = vec![];
    let is_prefill: Vec<bool> = vec![];

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
    assert_eq!(output.seq_ids.len(), 0);
    assert_eq!(output.next_tokens.len(), 0);
}

#[test]
fn test_model_single_token_batch() {
    let mut model = FakeModel::new(1000);
    let seq_ids = vec![1u64];
    let input_tokens = vec![vec![42u32]];
    let positions = vec![vec![0usize]];
    let kv_block_ids = vec![vec![0usize]];
    let num_computed_tokens = vec![0usize];
    let is_prefill = vec![true];

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
    assert_eq!(output.seq_ids.len(), 1);
    assert_eq!(output.next_tokens.len(), 1);
    assert_eq!(output.next_tokens[0], 1);
}

#[test]
fn test_model_large_batch() {
    let mut model = FakeModel::new(1000);
    let batch_size = 32;
    let seq_ids: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
    let input_tokens: Vec<Vec<u32>> = (0..batch_size).map(|i| vec![i as u32]).collect();
    let positions: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
    let kv_block_ids: Vec<Vec<usize>> = (0..batch_size).map(|_| vec![0]).collect();
    let num_computed_tokens: Vec<usize> = (0..batch_size).map(|_| 0).collect();
    let is_prefill: Vec<bool> = (0..batch_size).map(|_| true).collect();

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
    assert_eq!(output.seq_ids.len(), batch_size);
    assert_eq!(output.next_tokens.len(), batch_size);
}

#[test]
fn test_qwen3_config_default() {
    let config = Qwen3Config::default();
    assert_eq!(config.vocab_size(), 151936);
    assert_eq!(config.hidden_size(), 4096);
    assert_eq!(config.num_hidden_layers(), 32);
}

#[test]
fn test_qwen3_config_builder() {
    let config = Qwen3Config {
        vocab_size: Some(1000),
        hidden_size: Some(512),
        num_hidden_layers: Some(2),
        num_attention_heads: Some(8),
        num_key_value_heads: Some(2),
        intermediate_size: Some(1024),
        ..Default::default()
    };

    assert_eq!(config.vocab_size(), 1000);
    assert_eq!(config.hidden_size(), 512);
    assert_eq!(config.num_hidden_layers(), 2);
    assert_eq!(config.intermediate_size(), 1024);
}

#[test]
fn test_qwen3_config_text_config_fallback() {
    let text_config = vllm_model::qwen3_config::TextConfig {
        vocab_size: Some(5000),
        hidden_size: Some(256),
        num_hidden_layers: Some(4),
        num_attention_heads: Some(4),
        num_key_value_heads: Some(4),
        intermediate_size: Some(512),
        sliding_window: None,
        rope_theta: None,
        max_position_embeddings: None,
        rms_norm_eps: None,
    };
    let config = Qwen3Config {
        text_config: Some(text_config),
        ..Default::default()
    };

    assert_eq!(config.vocab_size(), 5000);
    assert_eq!(config.hidden_size(), 256);
    assert_eq!(config.num_hidden_layers(), 4);
}

#[test]
#[cfg(feature = "real_weights")]
fn test_qwen3_real_model_prefill() {
    use candle_core::Device;
    use vllm_model::loader::ModelLoader;

    let device = Device::Cpu;
    let loader = ModelLoader::builder(device)
        .with_model_dir("/models/Qwen3-0.6B".to_string())
        .with_kv_blocks(1024)
        .build()
        .expect("Failed to build loader");

    let mut model = loader.load_model().expect("Failed to load model");

    // Test prefill with simple tokens
    let tokens = vec![1u32, 2, 3, 4, 5];
    let positions: Vec<usize> = (0..tokens.len()).collect();

    let (logits, _) = model
        .forward_with_cache(&tokens, 0, &[0], &positions, true)
        .expect("Prefill failed");

    // Check logits shape
    assert_eq!(logits.dims().len(), 3, "logits should be 3D");
    assert_eq!(logits.dims()[0], 1, "batch size should be 1");
    assert_eq!(logits.dims()[1], 5, "seq_len should be 5");
    assert_eq!(logits.dims()[2], 151936, "vocab_size should be 151936");

    // Get next token from last position
    use candle_core::D;
    let next_token: u32 = logits
        .narrow(1, 4, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec0()
        .unwrap();

    println!("Next token from prefill: {}", next_token);
    assert!(next_token < 151936, "Token should be valid");
}

#[test]
#[cfg(feature = "real_weights")]
fn test_qwen3_real_model_decode() {
    use candle_core::Device;
    use vllm_model::loader::ModelLoader;

    let device = Device::Cpu;
    let loader = ModelLoader::builder(device)
        .with_model_dir("/models/Qwen3-0.6B".to_string())
        .with_kv_blocks(1024)
        .build()
        .expect("Failed to build loader");

    let mut model = loader.load_model().expect("Failed to load model");

    // Test decode
    let tokens = vec![42u32];
    let position = vec![5];

    let (logits, _) = model
        .forward_with_cache(&tokens, 5, &[0], &position, false)
        .expect("Decode failed");

    // Check logits shape - decode returns 2D [batch, vocab_size]
    assert_eq!(logits.dims().len(), 2, "decode logits should be 2D");
    assert_eq!(logits.dims()[0], 1, "batch size should be 1");
    assert_eq!(logits.dims()[1], 151936, "vocab_size should be 151936");

    // Get next token
    use candle_core::D;
    let next_token: u32 = logits
        .argmax(D::Minus1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec0()
        .unwrap();

    println!("Next token from decode: {}", next_token);
    assert!(next_token < 151936, "Token should be valid");
}
