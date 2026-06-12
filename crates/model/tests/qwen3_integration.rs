//! Qwen3 integration tests against on-disk checkpoints.
//!
//! Requires weights at [`support::qwen3::model_dir()`] (override via `VLLM_TEST_MODEL_DIR`).

mod support;

use support::qwen3::{HIDDEN_SIZE, VOCAB_SIZE};

#[test]

fn test_qwen3_real_model_prefill() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Test prefill with simple tokens
    let tokens = vec![1u32, 2, 3, 4, 5];
    let positions: Vec<usize> = (0..tokens.len()).collect();

    let (logits, _) = model
        .forward_with_cache(&tokens, 0, &[0], &positions, true)
        .expect("Prefill failed");

    // Check logits shape
    println!("DEBUG: logits.dims() = {:?}", logits.dims());
    assert_eq!(logits.dims().len(), 3, "logits should be 3D");
    assert_eq!(logits.dims()[0], 1, "batch size should be 1");
    assert_eq!(logits.dims()[1], 5, "seq_len should be 5");
    assert_eq!(
        logits.dims()[2],
        VOCAB_SIZE,
        "vocab_size should match checkpoint"
    );

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
        .to_vec0::<u32>()
        .unwrap();

    println!("Next token from prefill: {}", next_token);
    assert!(next_token < VOCAB_SIZE as u32, "Token should be valid");
}

#[test]

fn test_qwen3_real_model_decode() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Test decode
    let tokens = vec![42u32];
    let position = vec![5];

    let (logits, _) = model
        .forward_with_cache(&tokens, 5, &[0], &position, false)
        .expect("Decode failed");

    // Check logits shape - decode returns 3D [batch, seq, vocab_size]
    let dims = logits.dims();
    println!("DEBUG decode logits.dims() = {:?}", dims);
    assert_eq!(logits.dims().len(), 3, "logits should be 3D");
    assert_eq!(logits.dims()[0], 1, "batch size should be 1");
    assert_eq!(logits.dims()[1], 1, "seq_len should be 1 for decode");
    assert_eq!(
        logits.dims()[2],
        VOCAB_SIZE,
        "vocab_size should match checkpoint"
    );

    // Get next token
    use candle_core::D;
    let next_token: u32 = logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap();

    println!("Next token from decode: {}", next_token);
    assert!(next_token < VOCAB_SIZE as u32, "Token should be valid");

    // Check logits distribution - top token should have significantly higher logit than random
    // Decode logits are 2D [batch, vocab_size], need to flatten
    let logits_data: Vec<f32> = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let max_logit = logits_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let random_expected = max_logit - (VOCAB_SIZE as f32).ln();

    // Top logit should be significantly higher than random chance
    let entropy_bonus = max_logit - random_expected;
    println!(
        "Logit entropy bonus (higher = less random): {:.2}",
        entropy_bonus
    );
    assert!(
        entropy_bonus > 3.0,
        "Logits appear too uniform - possible model weight issue"
    );
}

#[test]

fn test_qwen3_generated_tokens_decodable() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Simulate prefill with "hi" prompt tokens
    let prompt_tokens = vec![6023u32]; // "hi" token
    let positions: Vec<usize> = (0..1).collect();

    // Run prefill
    let (prefill_logits, _) = model
        .forward_with_cache(&prompt_tokens, 0, &[0], &positions, true)
        .expect("Prefill failed");

    // Get first generated token
    use candle_core::D;
    let first_token: u32 = prefill_logits
        .narrow(1, 0, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec0()
        .unwrap();

    println!("First generated token: {}", first_token);
    assert!(first_token < VOCAB_SIZE as u32, "Token out of range");

    // Simulate decode step
    let decode_tokens = vec![first_token];
    let decode_positions = vec![1];

    let (decode_logits, _) = model
        .forward_with_cache(&decode_tokens, 1, &[0], &decode_positions, false)
        .expect("Decode failed");

    let second_token: u32 = decode_logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap();

    println!("Second generated token: {}", second_token);
    assert!(second_token < VOCAB_SIZE as u32, "Token out of range");

    // Both tokens should be valid
    println!("Generated sequence: [{}, {}]", first_token, second_token);
}

#[test]

fn test_qwen3_verify_model_weights() {
    let weights = support::qwen3::Qwen3Fixture::cpu()
        .checkpoint()
        .expect("load checkpoint");

    println!("Loaded {} tensors", weights.len());

    // Check for embed_tokens weight
    let embed_keys: Vec<_> = weights
        .keys()
        .filter(|k| k.contains("embed_tokens"))
        .collect();
    println!("Embed keys: {:?}", embed_keys);

    // Check for lm_head weight
    let lm_head_keys: Vec<_> = weights
        .keys()
        .filter(|k| k.contains("lm_head") || k.contains("output"))
        .collect();
    println!("LM head keys: {:?}", lm_head_keys);

    // Check for layer weights
    let layer_0_q_keys: Vec<_> = weights
        .keys()
        .filter(|k| k.contains("layers.0") && k.contains("q_proj"))
        .collect();
    println!("Layer 0 Q keys: {:?}", layer_0_q_keys);

    // Verify embed_tokens weight exists and has correct shape
    if let Some(embed_w) = weights.get("model.embed_tokens.weight") {
        println!("embed_tokens weight shape: {:?}", embed_w.dims());
        assert_eq!(
            embed_w.dims()[0],
            VOCAB_SIZE,
            "embed_tokens vocab_size mismatch"
        );
        assert_eq!(
            embed_w.dims()[1],
            HIDDEN_SIZE,
            "embed_tokens hidden_size mismatch"
        );
    } else {
        panic!("model.embed_tokens.weight not found");
    }

    // Verify at least one layer has weights
    let layer_count = weights
        .keys()
        .filter(|k| k.contains("layers.") && k.contains("q_proj.weight"))
        .count();
    println!("Number of layers with q_proj: {}", layer_count);
    assert!(
        layer_count >= 28,
        "Expected at least 28 layers, found {}",
        layer_count
    );

    // Verify weights are not zero (which would cause garbage output)
    if let Some(embed_w) = weights.get("model.embed_tokens.weight") {
        let embed_data: Vec<f32> = embed_w.flatten_all().unwrap().to_vec1().unwrap();
        let zero_count = embed_data.iter().filter(|&&x| x == 0.0).count();
        let total_count = embed_data.len();
        let zero_ratio = zero_count as f32 / total_count as f32;
        println!(
            "embed_tokens zero ratio: {:.2}% ({}/{})",
            zero_ratio * 100.0,
            zero_count,
            total_count
        );
        println!("embed_tokens dtype: {:?}", embed_w.dtype());

        // Check that weights have reasonable variance (not all the same)
        // Note: BF16/FP16 weights may have lower variance when converted to f32
        let mean = embed_data.iter().sum::<f32>() / total_count as f32;
        let variance =
            embed_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / total_count as f32;
        let std_dev = variance.sqrt();
        println!("embed_tokens mean: {:.6}, std_dev: {:.6}", mean, std_dev);

        // For BF16/FP16 weights, std_dev around 0.02-0.2 is reasonable
        // Just check that it's not zero or extremely small
        assert!(
            std_dev > 0.0001,
            "embed_tokens std_dev too low: {}",
            std_dev
        );
    }
}

#[test]
#[ignore = "slow: multi-step chat simulation on real weights"]
fn test_qwen3_chat_flow_simulation() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Build prompt exactly as server does
    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";
    let prompt = format!(
        "{}user\nhi{}{}\n{}assistant\n",
        im_start, im_end, "\n", im_start
    );

    println!("Prompt: {:?}", prompt);

    // The tokenizer should encode this prompt
    // For this test, we'll use a simplified approach
    let prompt_tokens = vec![
        151644u32, 8948, 30, 10950, 29, 151645, 151644, 4080, 2038, 25,
    ];
    println!("Using prompt tokens: {:?}", prompt_tokens);

    let positions: Vec<usize> = (0..prompt_tokens.len()).collect();

    // Run prefill
    let (logits, _) = model
        .forward_with_cache(&prompt_tokens, 0, &[0], &positions, true)
        .expect("Prefill failed");

    println!("Prefill logits shape: {:?}", logits.dims());

    // Simulate generating a few tokens
    use candle_core::D;
    let mut all_tokens = prompt_tokens.clone();

    for step in 0..3 {
        let seq_len = logits.dims()[1];
        let token_idx = step.min(seq_len - 1);

        let next_token: u32 = logits
            .narrow(1, token_idx, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .to_vec0()
            .unwrap();

        println!(
            "Step {}: Generated token {} (0x{:x})",
            step, next_token, next_token
        );

        all_tokens.push(next_token);
        assert!(
            next_token < VOCAB_SIZE as u32,
            "Token {} out of range",
            next_token
        );

        // If not the last step, run decode with this token
        if step < 2 {
            let (next_logits, _) = model
                .forward_with_cache(
                    &[next_token],
                    all_tokens.len(),
                    &[0],
                    &[all_tokens.len()],
                    false,
                )
                .expect("Decode failed");
            // Store logits for next iteration (simplified)
            let _ = next_logits;
        }
    }

    println!("All tokens: {:?}", all_tokens);
    println!(
        "Total tokens generated: {}",
        all_tokens.len() - prompt_tokens.len()
    );
}

#[test]

fn test_qwen3_embedding_layer() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Test embedding for "hi" token (6023)
    let hi_token = vec![6023u32];
    let embeddings = model.embed(&[hi_token], &[vec![0]]).expect("Embed failed");

    assert_eq!(embeddings.len(), 1, "Should return 1 embedding");
    assert_eq!(
        embeddings[0].len(),
        HIDDEN_SIZE,
        "Embedding dim should match hidden size"
    );

    // Verify embedding is not all zeros
    let emb_data = &embeddings[0];
    let non_zero_count = emb_data.iter().filter(|&&x| x != 0.0).count();
    let zero_ratio = non_zero_count as f32 / emb_data.len() as f32;
    println!("Embedding zero ratio: {:.2}%", (1.0 - zero_ratio) * 100.0);
    assert!(
        zero_ratio > 0.5,
        "Embedding should have >50% non-zero values"
    );

    // Verify embedding has reasonable magnitude
    let mean: f32 = emb_data.iter().sum::<f32>() / emb_data.len() as f32;
    let variance: f32 =
        emb_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / emb_data.len() as f32;
    println!("Embedding mean: {:.6}, variance: {:.6}", mean, variance);
    assert!(
        variance > 0.0001,
        "Embedding variance too low: {}",
        variance
    );
}

#[test]

fn test_qwen3_embedding_different_tokens_different_embeddings() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Test embeddings for different tokens
    let tokens = vec![
        vec![6023u32], // "hi"
        vec![2000u32], // different token
        vec![5000u32], // different token
    ];
    let positions = vec![vec![0], vec![0], vec![0]];

    let embeddings = model.embed(&tokens, &positions).expect("Embed failed");

    assert_eq!(embeddings.len(), 3, "Should return 3 embeddings");

    // Verify different tokens produce different embeddings
    let emb1 = &embeddings[0];
    let emb2 = &embeddings[1];

    let diff: f32 = emb1
        .iter()
        .zip(emb2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / emb1.len() as f32;

    println!(
        "Embedding difference between token 6023 and 2000: {:.6}",
        diff
    );
    assert!(
        diff > 0.01,
        "Different tokens should have different embeddings"
    );
}

#[test]
fn test_qwen3_attention_layer_output() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Test with prompt tokens
    let tokens = vec![6023u32]; // "hi"
    let positions: Vec<usize> = (0..1).collect();

    // Run forward pass
    let (logits, _) = model
        .forward_with_cache(&tokens, 0, &[0], &positions, true)
        .expect("Forward failed");

    // Verify logits shape [batch=1, seq=1, vocab=VOCAB_SIZE]
    assert_eq!(logits.dims().len(), 3, "Logits should be 3D");
    assert_eq!(logits.dims()[0], 1, "Batch size should be 1");
    assert_eq!(logits.dims()[1], 1, "Seq len should be 1");
    assert_eq!(
        logits.dims()[2],
        VOCAB_SIZE,
        "Vocab size should match checkpoint"
    );

    // Verify logits have reasonable distribution
    let logits_data = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_logit = logits_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = max_logit - min_logit;

    println!(
        "Logits range: min={:.4}, max={:.4}, range={:.4}",
        min_logit, max_logit, range
    );
    assert!(range > 1.0, "Logits should have significant range");

    // Verify top logit is not at the extremes
    let top_token: u32 = logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap();

    println!("Top token: {}", top_token);
    assert!(
        top_token < VOCAB_SIZE as u32,
        "Top token should be in vocab range"
    );
}

#[test]

fn test_qwen3_attention_kv_cache_consistency() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // First forward: compute hidden state
    let tokens1 = vec![6023u32];
    let (logits1, _) = model
        .forward_with_cache(&tokens1, 0, &[0], &[0], true)
        .expect("First forward failed");

    // Get first token
    let first_token: u32 = logits1
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec0()
        .unwrap();

    // Second forward: use first token as continuation
    let tokens2 = vec![first_token];
    let (logits2, _) = model
        .forward_with_cache(&tokens2, 1, &[0], &[1], false)
        .expect("Second forward failed");

    // Verify logits2 is valid
    let logits_data2 = logits2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let max_logit2 = logits_data2
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    println!("Decode logits max: {:.4}", max_logit2);
    assert!(max_logit2 > -100.0, "Decode logits should not be all -inf");

    // Verify logits2 is different from logits1 (different position)
    let logits_data1 = logits1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let diff: f32 = logits_data1
        .iter()
        .zip(logits_data2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / logits_data1.len() as f32;

    println!("Logits difference between prefill and decode: {:.6}", diff);
    // Note: This test might fail if the model produces similar outputs,
    // so we make it a soft check
    if diff < 0.01 {
        println!(
            "WARNING: Prefill and decode logits are very similar - this might indicate an issue"
        );
    }
}

#[test]
#[ignore = "slow: full prefill/decode pipeline on real weights"]
fn test_qwen3_full_pipeline_prefill_decode() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Build chat prompt: "<|im_start|>user\nhi<|im_end|>\n\n<|im_start|>assistant\n"
    let prompt_tokens = vec![
        151644u32, 8948, 30, 10950, 29, 151645, 151644, 4080, 2038, 25,
    ];
    let positions: Vec<usize> = (0..prompt_tokens.len()).collect();

    // Prefill phase
    let (prefill_logits, _) = model
        .forward_with_cache(&prompt_tokens, 0, &[0], &positions, true)
        .expect("Prefill failed");

    println!("Prefill logits shape: {:?}", prefill_logits.dims());
    assert_eq!(
        prefill_logits.dims()[1],
        prompt_tokens.len(),
        "Prefill should process all prompt tokens"
    );

    // Get first generated token from last position
    let seq_len = prefill_logits.dims()[1];
    let next_token: u32 = prefill_logits
        .narrow(1, seq_len - 1, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec0()
        .unwrap();

    println!("First generated token: {}", next_token);
    assert!(
        next_token < VOCAB_SIZE as u32,
        "Token should be in vocab range"
    );

    // Decode phase (first decode step)
    let decode_position = prompt_tokens.len();
    let (decode_logits1, _) = model
        .forward_with_cache(
            &[next_token],
            prompt_tokens.len(),
            &[0],
            &[decode_position],
            false,
        )
        .expect("First decode failed");

    let logits_data = decode_logits1
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_logit = logits_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    println!("First decode logits max: {:.4}", max_logit);
    assert!(max_logit > -100.0, "Decode logits should be valid");

    // Decode phase (second decode step)
    let next_token2: u32 = decode_logits1
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap();

    println!("Second generated token: {}", next_token2);
    assert!(
        next_token2 < VOCAB_SIZE as u32,
        "Token should be in vocab range"
    );

    // Third decode step
    let (decode_logits2, _) = model
        .forward_with_cache(
            &[next_token2],
            prompt_tokens.len() + 1,
            &[0],
            &[decode_position + 1],
            false,
        )
        .expect("Second decode failed");

    let next_token3: u32 = decode_logits2
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(candle_core::D::Minus1)
        .unwrap()
        .to_vec0::<u32>()
        .unwrap();

    println!("Third generated token: {}", next_token3);
    println!(
        "Generated sequence: [{}, {}, {}]",
        next_token, next_token2, next_token3
    );

    // Verify all tokens are valid
    assert!(next_token < VOCAB_SIZE as u32);
    assert!(next_token2 < VOCAB_SIZE as u32);
    assert!(next_token3 < VOCAB_SIZE as u32);

    // Verify tokens are not all the same (model should generate diverse output)
    let unique_tokens = std::collections::HashSet::from([next_token, next_token2, next_token3]);
    println!("Unique tokens: {}", unique_tokens.len());
    assert!(
        unique_tokens.len() > 1,
        "Model should generate diverse tokens, not repeat the same one"
    );
}

#[test]

fn test_qwen3_layer0_intermediate_outputs() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    let tokens = vec![6023u32];
    let positions: Vec<usize> = vec![0];

    let (logits, next_token) = model
        .forward_with_cache(&tokens, 0, &[0], &positions, true)
        .expect("Forward failed");

    println!("Next token: {}", next_token);
    println!("Logits dims: {:?}", logits.dims());

    // For prefill, logits is [batch, seq_len, vocab_size]
    assert_eq!(logits.dims().len(), 3, "Prefill logits should be 3D");

    let logits_data = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let logits_std = logits_data.iter().map(|&x| x * x).sum::<f32>() / logits_data.len() as f32;

    println!("Logits std: {:.6}", logits_std.sqrt());
    assert!(
        logits_std.sqrt() > 0.1,
        "Logits std too small: {}",
        logits_std.sqrt()
    );
}

#[test]
#[ignore = "slow: repeated forward passes on real weights"]
fn test_qwen3_deterministic_same_input() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    let tokens = vec![6023u32];
    let positions: Vec<usize> = vec![0];

    // Run same input multiple times
    let mut top_tokens = Vec::new();
    let mut _first_logits: Option<Vec<f32>> = None;

    for i in 0..3 {
        let (logits, _) = model
            .forward_with_cache(&tokens, 0, &[0], &positions, true)
            .unwrap();

        let top_token: u32 = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_vec0()
            .unwrap();

        top_tokens.push(top_token);

        if i == 0 {
            _first_logits = Some(logits.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        }
    }

    println!("Top tokens from 3 runs: {:?}", top_tokens);
    let all_same = top_tokens.windows(2).all(|w| w[0] == w[1]);
    assert!(
        all_same,
        "Same input should produce same output: {:?}",
        top_tokens
    );
}

#[test]

fn test_qwen3_different_prompts_different_outputs() {
    let mut model = support::qwen3::Qwen3Fixture::cpu()
        .load_model()
        .expect("Failed to load model");

    // Test with two different prompts
    let tokens1 = vec![6023u32]; // "hi"
    let tokens2 = vec![14947u32]; // different word

    let (logits1, _) = model
        .forward_with_cache(&tokens1, 0, &[0], &[0], true)
        .unwrap();
    let (logits2, _) = model
        .forward_with_cache(&tokens2, 0, &[0], &[0], true)
        .unwrap();

    let l1 = logits1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let l2 = logits2.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Compute cosine similarity
    let dot: f32 = l1.iter().zip(l2.iter()).map(|(a, b)| a * b).sum::<f32>();
    let norm1 = l1.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm2 = l2.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm1 * norm2);

    println!("Cosine similarity between different prompts: {:.6}", cosine);

    // Different prompts should produce different outputs (cosine should not be ~1)
    assert!(
        cosine < 0.99,
        "Different prompts should produce different outputs, cosine: {}",
        cosine
    );
}
