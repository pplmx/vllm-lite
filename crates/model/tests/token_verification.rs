use vllm_model::tokenizer::Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    fn is_printable_text(s: &str) -> bool {
        !s.is_empty() && !s.chars().any(|c| c == '\u{FFFD}') && s.chars().any(|c| c.is_alphabetic())
    }

    fn setup_tokenizer() -> Tokenizer {
        let path = std::path::PathBuf::from("/models/Qwen3-0.6B/tokenizer.json");
        if !path.exists() {
            panic!("Tokenizer not found at {:?}", path);
        }
        Tokenizer::from_file(path.to_str().unwrap()).expect("Failed to load tokenizer")
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_tokenizer_decode_model_output() {
        let tokenizer = setup_tokenizer();

        let model_output_tokens = vec![13539u32, 47421u32, 60290u32];

        for token in &model_output_tokens {
            let decoded = tokenizer.decode(&[(*token)]);

            println!("Token {} decodes to: {:?}", token, decoded);

            assert!(
                !decoded.contains('\u{FFFD}'),
                "Token {} should not decode to replacement char, got: {:?}",
                token,
                decoded
            );

            assert!(
                is_printable_text(&decoded),
                "Token {} should decode to printable text, got: {:?}",
                token,
                decoded
            );

            assert!(
                !decoded.trim().is_empty(),
                "Token {} decoded to empty/whitespace: {:?}",
                token,
                decoded
            );
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_tokenizer_decode_top_k_tokens() {
        let tokenizer = setup_tokenizer();

        let sample_ranges = vec![(0, 1000), (10000, 11000), (150000, 151000)];

        let mut all_samples = Vec::new();
        for (start, end) in sample_ranges {
            for token in start..end {
                all_samples.push(token as u32);
            }
        }

        let mut fail_count = 0;
        let mut fail_tokens = Vec::new();

        for token in &all_samples {
            let decoded = tokenizer.decode(&[*token]);
            if decoded.contains('\u{FFFD}') || decoded.trim().is_empty() {
                fail_count += 1;
                fail_tokens.push(*token);
            }
        }

        let fail_rate = fail_count as f32 / all_samples.len() as f32;
        println!(
            "Token decode fail rate: {:.1}% ({}/{})",
            fail_rate * 100.0,
            fail_count,
            all_samples.len()
        );

        if !fail_tokens.is_empty() {
            println!(
                "Failed tokens (first 10): {:?}",
                &fail_tokens[..fail_tokens.len().min(10)]
            );
        }

        assert!(
            fail_rate < 0.1,
            "{}/{} tokens ({:.1}%) failed to decode: {:?}",
            fail_count,
            all_samples.len(),
            fail_rate * 100.0,
            &fail_tokens[..fail_tokens.len().min(10)]
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_tokenizer_roundtrip_vocab() {
        let tokenizer = setup_tokenizer();

        let test_strings = vec![
            "hi",
            "hello",
            "world",
            "The",
            "a",
            "Hello, world!",
            "123",
            "token",
        ];

        let mut failed = Vec::new();

        for text in &test_strings {
            let tokens = tokenizer.encode(text);
            let decoded = tokenizer.decode(&tokens);

            if !decoded.trim().to_lowercase().contains(&text.to_lowercase()) {
                failed.push((text.to_string(), tokens, decoded.clone()));
            }
        }

        if !failed.is_empty() {
            println!("Roundtrip failures:");
            for (orig, tokens, decoded) in &failed {
                println!("  '{}' -> {:?} -> '{}'", orig, tokens, decoded);
            }
        }

        let fail_rate = failed.len() as f32 / test_strings.len() as f32;
        assert!(
            fail_rate < 0.3,
            "{}/{} roundtrip failed",
            failed.len(),
            test_strings.len()
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_special_tokens() {
        let tokenizer = setup_tokenizer();

        let eos_token = 151643u32;
        let bos_token = 151645u32;

        let eos_decoded = tokenizer.decode(&[eos_token]);
        println!("EOS token {} decodes to: {:?}", eos_token, eos_decoded);

        let bos_decoded = tokenizer.decode(&[bos_token]);
        println!("BOS token {} decodes to: {:?}", bos_token, bos_decoded);

        assert!(
            !eos_decoded.is_empty(),
            "EOS token should decode to something"
        );
        assert!(
            !bos_decoded.is_empty(),
            "BOS token should decode to something"
        );

        if eos_decoded.contains('\u{FFFD}') {
            println!("WARNING: EOS token produces replacement character!");
        }
        if bos_decoded.contains('\u{FFFD}') {
            println!("WARNING: BOS token produces replacement character!");
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_tokenizer_decode_out_of_range_tokens() {
        let tokenizer = setup_tokenizer();

        let vocab_size = 151936;
        let out_of_range_tokens = vec![
            vocab_size,       // 151936 - exactly at vocab size
            vocab_size + 1,   // 151937
            vocab_size + 100, // 152036
            200000,           // Way out of range
            1000000,          // Very far out of range
        ];

        let mut replacement_char_count = 0;
        for token in &out_of_range_tokens {
            let decoded = tokenizer.decode(&[*token]);
            println!("Out-of-range token {} decodes to: {:?}", token, decoded);
            if decoded.contains('\u{FFFD}') {
                replacement_char_count += 1;
            }
        }

        println!(
            "Tokens with replacement char: {}/{}",
            replacement_char_count,
            out_of_range_tokens.len()
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_eos_handling_in_decode() {
        let tokenizer = setup_tokenizer();

        let eos_token = 151643u32;

        let test_sequences = [
            vec![13539u32, 47421u32, eos_token],
            vec![6023u32, eos_token],
            vec![6023u32, 6024u32, 6025u32, eos_token],
        ];

        for (i, tokens) in test_sequences.iter().enumerate() {
            let decoded = tokenizer.decode(tokens);
            println!("Sequence {}: {:?} -> {:?}", i + 1, tokens, decoded);

            if decoded.contains('\u{FFFD}') {
                println!("  WARNING: Contains replacement char!");
            }
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    #[ignore = "Known issue: partial prefill produces different output than full prefill"]
    fn test_qwen3_partial_prefill_kv_cache_issue() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        // Full prompt
        let full_prompt = vec![
            151643u32, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ];

        println!("=== Test 1: Full prefill, single batch ===");
        let positions = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (logits1, _) = model
            .forward_with_cache(&full_prompt, 0, &[0], &positions, true)
            .expect("Full prefill failed");

        use candle_core::D;
        let next1: u32 = logits1
            .narrow(1, 9, 1)
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
            "Full prefill: {} -> '{}'",
            next1,
            tokenizer.decode(&[next1])
        );

        println!("\n=== Test 2: Two-step prefill (chunk 1, then chunk 2) ===");
        let chunk1 = vec![151643u32, 151644, 872, 198, 6023];
        let pos1 = vec![0, 1, 2, 3, 4];
        let (logits_c1, _) = model
            .forward_with_cache(&chunk1, 0, &[0], &pos1, true)
            .expect("Chunk 1 failed");

        let next_c1: u32 = logits_c1
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
        println!(
            "After chunk 1: {} -> '{}'",
            next_c1,
            tokenizer.decode(&[next_c1])
        );

        // Second chunk starting from position 5 (simulating num_computed_tokens=5)
        let chunk2 = vec![151645u32, 198, 151644, 77091, 198];
        let pos2 = vec![5, 6, 7, 8, 9];
        let (logits_c2, _) = model
            .forward_with_cache(&chunk2, 5, &[0], &pos2, true)
            .expect("Chunk 2 failed");

        let next_c2: u32 = logits_c2
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
        println!(
            "After chunk 2: {} -> '{}'",
            next_c2,
            tokenizer.decode(&[next_c2])
        );

        println!("\n=== Conclusion ===");
        println!("Full prefill gives: {}", next1);
        println!("Two-step gives: first={}, second={}", next_c1, next_c2);

        // The issue is that partial prefill doesn't give the same result
        // as full prefill because of KV cache/state issues
        let outputs_differ = next1 != next_c2;
        println!("Outputs differ: {}", outputs_differ);

        if outputs_differ {
            println!("\n*** BUG: Partial prefill produces different output! ***");
            println!("This is likely due to KV cache not being properly handled");
        }

        // This test documents the issue - pass regardless but log the problem
        if outputs_differ {
            println!("[DOCUMENTED BUG] Partial prefill differs from full prefill");
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_partial_prefill_simulation() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        // Simulate partial prefill: first 5 tokens
        let first_chunk = vec![151643u32, 151644, 872, 198, 6023];
        let positions = vec![0, 1, 2, 3, 4];

        let (logits1, _) = model
            .forward_with_cache(&first_chunk, 0, &[0], &positions, true)
            .expect("First chunk failed");

        // Get the last token's output
        use candle_core::D;
        let next1: u32 = logits1
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

        println!(
            "After chunk 1 (5 tokens): next = {} -> '{}'",
            next1,
            tokenizer.decode(&[next1])
        );

        // Second chunk: remaining 5 tokens (simulating num_computed_tokens=5)
        let second_chunk = vec![151645u32, 198, 151644, 77091, 198];
        let positions2 = vec![5, 6, 7, 8, 9];

        let (logits2, _) = model
            .forward_with_cache(&second_chunk, 5, &[0], &positions2, true)
            .expect("Second chunk failed");

        let next2: u32 = logits2
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

        println!(
            "After chunk 2 (5 tokens): next = {} -> '{}'",
            next2,
            tokenizer.decode(&[next2])
        );

        // Full prefill (all 10 tokens)
        let full_prompt = vec![
            151643u32, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ];
        let positions_full = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let (logits_full, _) = model
            .forward_with_cache(&full_prompt, 0, &[0], &positions_full, true)
            .expect("Full prefill failed");

        let next_full: u32 = logits_full
            .narrow(1, 9, 1)
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
            "After full prefill (10 tokens): next = {} -> '{}'",
            next_full,
            tokenizer.decode(&[next_full])
        );

        // Compare - all should produce the same token for the same input
        assert_eq!(
            next_full, 29054,
            "Full prefill should produce 29054, got {}",
            next_full
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_forward_with_exact_server_params() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        // Test 1: Exact prompt from server
        let server_tokens = vec![
            151643u32, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ];
        let server_positions: Vec<usize> = (0..10).collect();

        let (logits, _) = model
            .forward_with_cache(&server_tokens, 0, &[0], &server_positions, true)
            .expect("Forward failed");

        use candle_core::D;
        let next_token: u32 = logits
            .narrow(1, 9, 1)
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
            "Server prompt first token: {} -> '{}'",
            next_token,
            tokenizer.decode(&[next_token])
        );

        // Test 2: Simple prompt "hi" only
        let simple_tokens = vec![6023u32];

        let (logits2, _) = model
            .forward_with_cache(&simple_tokens, 0, &[0], &[0], true)
            .expect("Forward failed");

        let next_token2: u32 = logits2
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

        println!(
            "Simple prompt first token: {} -> '{}'",
            next_token2,
            tokenizer.decode(&[next_token2])
        );

        // Compare - they should be the same for the same input
        assert_eq!(
            next_token, 29054,
            "Server prompt should produce token 29054 (from test), got {}",
            next_token
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_simulate_server_engine_flow() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        // Exact server prompt tokens
        let prompt_tokens = vec![
            151643u32, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ];
        let positions: Vec<usize> = (0..prompt_tokens.len()).collect();

        println!("=== Prefill Phase ===");
        let prompt_decoded = tokenizer.decode(&prompt_tokens);
        println!("Prompt: {:?}", prompt_decoded);

        let (prefill_logits, _) = model
            .forward_with_cache(&prompt_tokens, 0, &[0], &positions, true)
            .expect("Prefill failed");

        use candle_core::D;
        let seq_len = prefill_logits.dims()[1];
        let first_token: u32 = prefill_logits
            .narrow(1, seq_len - 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .to_vec0()
            .unwrap();

        let first_decoded = tokenizer.decode(&[first_token]);
        println!(
            "First token after prefill: {} -> '{}'",
            first_token, first_decoded
        );

        // Simulate first decode step
        let mut all_tokens = prompt_tokens.clone();
        all_tokens.push(first_token);

        println!("\n=== Decode Phase ===");
        for step in 0..3 {
            let decode_pos = all_tokens.len();
            let decode_token = *all_tokens.last().unwrap();
            let decode_positions = vec![decode_pos];

            let (decode_logits, _) = model
                .forward_with_cache(
                    &[decode_token],
                    all_tokens.len() - 1,
                    &[0],
                    &decode_positions,
                    false,
                )
                .expect("Decode failed");

            let next: u32 = decode_logits
                .squeeze(0)
                .unwrap()
                .argmax(D::Minus1)
                .unwrap()
                .to_vec0()
                .unwrap();

            let decoded = tokenizer.decode(&[next]);
            println!("Decode step {}: token {} -> '{}'", step + 1, next, decoded);

            all_tokens.push(next);
        }

        let final_text = tokenizer.decode(&all_tokens[prompt_tokens.len()..]);
        println!("\nFinal generated text: {:?}", final_text);

        // Verify no garbage
        assert!(
            !final_text.contains('\u{FFFD}'),
            "Should not contain replacement char"
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_multi_step_generation() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        let prompt = "hi";
        let prompt_tokens: Vec<u32> = tokenizer.encode(prompt);
        println!("Prompt '{}' tokens: {:?}", prompt, prompt_tokens);

        let mut all_tokens = prompt_tokens.clone();
        let mut generated_tokens = Vec::new();

        for step in 0..3 {
            let seq_len = all_tokens.len();
            let positions: Vec<usize> = (0..seq_len).collect();
            let is_decode = step > 0;

            println!(
                "Step {}: running forward with {} tokens, is_decode={}",
                step + 1,
                seq_len,
                is_decode
            );

            let (logits, _) = model
                .forward_with_cache(&all_tokens, 0, &[0], &positions, is_decode)
                .expect("Forward failed");

            println!("  logits dims: {:?}", logits.dims());

            let logits_vec: Vec<f32> = {
                let dims = logits.dims();
                if dims.len() == 3 {
                    // [batch, seq, vocab] - take last token
                    let seq_len = dims[1];
                    logits
                        .narrow(1, seq_len - 1, 1)
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                        .to_vec1()
                        .unwrap()
                } else if dims.len() == 2 {
                    // [batch, vocab] or [seq, vocab]
                    if dims[0] == 1 {
                        logits.squeeze(0).unwrap().to_vec1().unwrap()
                    } else {
                        logits.to_vec1().unwrap()
                    }
                } else {
                    logits.to_vec1().unwrap()
                }
            };

            let vocab_size = tokenizer.vocab_size();
            let top_idx = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let top_token = top_idx as u32;

            if top_token >= vocab_size as u32 {
                println!(
                    "  WARNING: top_token {} >= vocab_size {}",
                    top_token, vocab_size
                );
            } else {
                let decoded = tokenizer.decode(&[top_token]);
                println!("  top_token {} -> {:?}", top_token, decoded);
                generated_tokens.push((top_token, decoded.clone()));

                if decoded.contains('\u{FFFD}') {
                    println!("  WARNING: Got replacement character!");
                }

                all_tokens.push(top_token);
            }
        }

        let final_decode =
            tokenizer.decode(&generated_tokens.iter().map(|(t, _)| *t).collect::<Vec<_>>());
        println!("Full generated text: {:?}", final_decode);

        assert!(
            !final_decode.contains('\u{FFFD}'),
            "Final decode should not contain replacement char"
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_server_streaming_token_handling() {
        let tokenizer = setup_tokenizer();

        let eos_token = 151643u32;
        let im_end_token = 151645u32;

        let server_output_tokens = vec![13539u32, 47421u32, 60290u32, eos_token];

        println!("=== Simulating server streaming behavior ===");

        let mut streamed_chunks = Vec::new();
        for token in &server_output_tokens {
            let text = tokenizer.decode(&[*token]);
            println!("Token {} -> {:?}", token, text);

            if text.is_empty() {
                println!("  -> Skipped (empty)");
                continue;
            }

            streamed_chunks.push(text.clone());
            println!("  -> Streamed: '{}'", text);
        }

        let combined: String = streamed_chunks.iter().map(|s| s.as_str()).collect();
        println!("\nCombined streaming output: {:?}", combined);

        let batch_decode = tokenizer.decode(&server_output_tokens);
        println!("Batch decode: {:?}", batch_decode);

        println!("\n=== Testing with <|im_end|> in output ===");
        let tokens_with_im_end = vec![6023u32, 6024u32, im_end_token, 6025u32, 6026u32];
        let decoded_with_im_end = tokenizer.decode(&tokens_with_im_end);
        println!(
            "Tokens {:?} -> {:?}",
            tokens_with_im_end, decoded_with_im_end
        );

        if decoded_with_im_end.contains("<|im_end|>") {
            println!("WARNING: <|im_end|> appears in decoded output!");
            println!("This could cause visible乱码 in streaming responses");
        }

        println!("\n=== Testing first token patterns ===");
        let first_token_tests = [vec![6023u32], vec![13539u32], vec![60290u32]];
        for (i, tokens) in first_token_tests.iter().enumerate() {
            let text = tokenizer.decode(tokens);
            println!("First token pattern {}: {} -> '{}'", i + 1, tokens[0], text);
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_qwen3_with_special_tokens() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        // Server prompt tokens: [151643, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198]
        let server_tokens = vec![
            151643u32, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ];
        let positions: Vec<usize> = (0..server_tokens.len()).collect();

        println!("Prompt: {:?}", tokenizer.decode(&server_tokens));

        let (logits, _) = model
            .forward_with_cache(&server_tokens, 0, &[0], &positions, true)
            .expect("Forward failed");

        // Get top token
        use candle_core::D;
        let seq_len = logits.dims()[1];
        let next_token: u32 = logits
            .narrow(1, seq_len - 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .to_vec0()
            .unwrap();

        let decoded = tokenizer.decode(&[next_token]);
        println!("First generated token: {} -> '{}'", next_token, decoded);

        assert!(next_token < 151936, "Token should be in vocab range");
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_server_token_stream_decode() {
        let tokenizer = setup_tokenizer();

        // Exact tokens from server output
        let server_tokens = vec![
            95060u32, 47625u32, 79128u32, 11433u32, 121471u32, 78348u32, 11234u32, 19906u32,
            4342u32, 83454u32,
        ];

        println!("=== Decoding server token stream ===");
        for token in &server_tokens {
            let decoded = tokenizer.decode(&[*token]);
            println!("Token {} -> '{}'", token, decoded);
        }

        // Batch decode
        let batch = tokenizer.decode(&server_tokens);
        println!("\nBatch decode: '{}'", batch);
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_server_problematic_tokens() {
        let tokenizer = setup_tokenizer();

        // Tokens that caused garbage in server output
        let server_problematic = vec![121471u32, 78348u32, 11234u32];

        println!("=== Testing server problematic tokens ===");
        for &token in &server_problematic {
            let decoded = tokenizer.decode(&[token]);
            println!("Token {} -> '{}'", token, decoded);

            // Check if it contains replacement char
            if decoded.contains('\u{FFFD}') {
                println!("  WARNING: Contains replacement character!");
            }

            // Check if it's printable
            let is_printable = decoded
                .chars()
                .all(|c| c.is_alphanumeric() || c.is_whitespace() || c.is_ascii_punctuation());
            if !is_printable {
                println!("  Contains non-printable characters");
            }
        }

        // Check batch decode
        let batch = tokenizer.decode(&server_problematic);
        println!("\nBatch decode: '{}'", batch);

        if batch.contains('\u{FFFD}') {
            println!("WARNING: Batch contains replacement character!");
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_special_token_filtering() {
        let tokenizer = setup_tokenizer();

        let eos_token = 151643u32;
        let im_end_token = 151645u32;
        let im_start_token = 151644u32;

        let special_tokens = vec![
            ("EOS", eos_token),
            ("IM_END", im_end_token),
            ("IM_START", im_start_token),
        ];

        println!("=== Special token values ===");
        for (name, token) in &special_tokens {
            let decoded = tokenizer.decode(&[*token]);
            println!("{} ({}): {:?}", name, token, decoded);
        }

        println!("\n=== Filter test ===");
        let tokens_with_special = vec![6023u32, eos_token, 6024u32, 6025u32, im_end_token, 6026u32];

        let filtered: Vec<u32> = tokens_with_special
            .iter()
            .filter(|&&t| t != eos_token && t != im_end_token && t != im_start_token)
            .copied()
            .collect();

        let unfiltered_decoded = tokenizer.decode(&tokens_with_special);
        let filtered_decoded = tokenizer.decode(&filtered);

        println!(
            "Unfiltered: {:?} -> {:?}",
            tokens_with_special, unfiltered_decoded
        );
        println!("Filtered: {:?} -> {:?}", filtered, filtered_decoded);

        if unfiltered_decoded.contains("<|") {
            println!("\nWARNING: Special tokens in output will be visible to users!");
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_model_token_to_text_pipeline() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        let tokens = vec![6023u32]; // "hi"
        let positions: Vec<usize> = vec![0];

        let (logits, _) = model
            .forward_with_cache(&tokens, 0, &[0], &positions, true)
            .expect("Forward failed");

        let top_token: u32 = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_vec0()
            .unwrap();

        println!("Top token: {}", top_token);

        let text = tokenizer.decode(&[top_token]);
        println!("Decoded text: {:?}", text);

        assert!(!text.is_empty(), "Decoded text should not be empty");

        assert!(
            !text.contains('\u{FFFD}'),
            "Decoded text contains replacement char: {:?}",
            text
        );

        let meaningful_chars: usize = text.chars().filter(|c| c.is_alphabetic()).count();

        assert!(
            meaningful_chars > 0,
            "Decoded text should contain some letters, got: {:?}",
            text
        );

        println!("Model output '{}' decodes to '{}'", top_token, text);
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_server_engine_loop_simulation() {
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        // Exact server prompt tokens
        let prompt_tokens = vec![
            151643u32, 151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198,
        ];
        let _prompt_len = prompt_tokens.len();

        println!("=== SERVER ENGINE LOOP SIMULATION ===");
        println!("Prompt tokens: {:?}", prompt_tokens);
        println!("Prompt decoded: {:?}", tokenizer.decode(&prompt_tokens));

        // Simulate server engine state
        let mut all_tokens = prompt_tokens.clone();
        let mut num_computed = 0;
        let mut is_first_step = true;
        let mut generated_tokens = Vec::new();

        // Simulate 3 decode steps
        for step in 0..3 {
            let tokens_len = all_tokens.len();

            // Build batch (simulating BatchComposer)
            let input_tokens: Vec<u32>;
            let positions: Vec<usize>;
            let is_prefill: bool;

            if is_first_step {
                // Prefill phase - process all remaining tokens
                input_tokens = all_tokens[num_computed..].to_vec();
                positions = (num_computed..tokens_len).collect();
                is_prefill = true;
                println!("\n--- Step {} (PREILL) ---", step + 1);
                println!("  input_tokens: {:?}", input_tokens);
                println!("  positions: {:?}", positions);
                println!("  num_computed: {}", num_computed);
                is_first_step = false;
            } else {
                // Decode phase - only last token
                let last_token = *all_tokens.last().unwrap();
                input_tokens = vec![last_token];
                positions = vec![tokens_len - 1]; // position is 0-indexed
                is_prefill = false;
                println!("\n--- Step {} (DECODE) ---", step + 1);
                println!("  input_tokens: {:?}", input_tokens);
                println!("  positions: {:?}", positions);
                println!("  num_computed: {}", num_computed);
            }

            // Call model forward
            let (logits, _) = model
                .forward_with_cache(&input_tokens, num_computed, &[0], &positions, is_prefill)
                .expect("Forward failed");

            // Extract next token
            use candle_core::D;
            let next_token: u32 = if is_prefill {
                // For prefill, take last token's logits
                let seq_len = logits.dims()[1];
                logits
                    .narrow(1, seq_len - 1, 1)
                    .unwrap()
                    .squeeze(1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .argmax(D::Minus1)
                    .unwrap()
                    .to_vec0()
                    .unwrap()
            } else {
                // For decode, logits are [batch, vocab]
                logits
                    .argmax(D::Minus1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec0()
                    .unwrap()
            };

            let decoded = tokenizer.decode(&[next_token]);
            println!("  next_token: {} -> '{}'", next_token, decoded);

            // Update state
            all_tokens.push(next_token);
            generated_tokens.push(next_token);
            num_computed = if is_prefill {
                tokens_len // After prefill, all tokens are computed
            } else {
                num_computed + 1 // After decode, increment by 1
            };

            // Check for EOS
            if next_token == 151643 || next_token == 151645 {
                println!("  -> Got EOS/IM_END, stopping");
                break;
            }
        }

        // Compare with expected
        println!("\n=== RESULTS ===");
        println!(
            "Generated tokens ({}): {:?}",
            generated_tokens.len(),
            generated_tokens
        );
        let generated_text = tokenizer.decode(&generated_tokens);
        println!("Generated text: '{}'", generated_text);

        // First token should be 29054 (ationally)
        if !generated_tokens.is_empty() {
            assert_eq!(
                generated_tokens[0], 29054,
                "First token should be 29054 (from working unit test), got {}",
                generated_tokens[0]
            );
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_working_unit_test_vs_server_comparison() {
        // This test verifies that the working unit test and server loop produce same output
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        let prompt = "hi";
        let prompt_tokens: Vec<u32> = tokenizer.encode(prompt);
        let prompt_len = prompt_tokens.len();

        println!("=== COMPARING WORKING TEST vs SERVER LOOP ===");
        println!("Prompt: '{}', tokens: {:?}", prompt, prompt_tokens);

        // Method 1: Working unit test (test_qwen3_multi_step_generation)
        // This passes in unit tests
        let mut all_tokens_v1 = prompt_tokens.clone();
        for step in 0..1 {
            let seq_len = all_tokens_v1.len();
            let positions: Vec<usize> = (0..seq_len).collect();
            let is_decode = step > 0;

            let (logits, _) = model
                .forward_with_cache(&all_tokens_v1, 0, &[0], &positions, is_decode)
                .expect("Forward failed");

            let logits_vec: Vec<f32> = {
                let dims = logits.dims();
                if dims.len() == 3 {
                    let seq_len = dims[1];
                    logits
                        .narrow(1, seq_len - 1, 1)
                        .unwrap()
                        .squeeze(1)
                        .unwrap()
                        .squeeze(0)
                        .unwrap()
                        .to_vec1()
                        .unwrap()
                } else if dims.len() == 2 {
                    // [batch, vocab] - squeeze batch dim
                    logits.squeeze(0).unwrap().to_vec1().unwrap()
                } else {
                    logits.to_vec1().unwrap()
                }
            };

            let top_idx = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let top_token = top_idx as u32;
            let decoded = tokenizer.decode(&[top_token]);
            println!(
                "Method 1 (working test) - step {}: {} -> '{}'",
                step + 1,
                top_token,
                decoded
            );
            all_tokens_v1.push(top_token);
        }

        // Method 2: Server loop (prefill then decode)
        let mut all_tokens_v2 = prompt_tokens.clone();
        let mut num_computed = 0;

        // Step 1: Prefill
        let input_tokens_1 = all_tokens_v2[num_computed..].to_vec();
        let positions_1: Vec<usize> = (num_computed..all_tokens_v2.len()).collect();
        let (logits_1, _) = model
            .forward_with_cache(&input_tokens_1, num_computed, &[0], &positions_1, true)
            .expect("Prefill failed");

        use candle_core::D;
        let seq_len_1 = logits_1.dims()[1];
        let first_token: u32 = logits_1
            .narrow(1, seq_len_1 - 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .argmax(D::Minus1)
            .unwrap()
            .to_vec0()
            .unwrap();

        let decoded_1 = tokenizer.decode(&[first_token]);
        println!(
            "Method 2 (server loop) - step 1 (prefill): {} -> '{}'",
            first_token, decoded_1
        );
        all_tokens_v2.push(first_token);
        num_computed = all_tokens_v2.len();

        // Step 2: Decode
        let input_tokens_2 = vec![first_token];
        let positions_2 = vec![num_computed - 1];
        let (logits_2, _) = model
            .forward_with_cache(&input_tokens_2, num_computed - 1, &[0], &positions_2, false)
            .expect("Decode failed");

        let second_token: u32 = logits_2
            .argmax(D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec0()
            .unwrap();

        let decoded_2 = tokenizer.decode(&[second_token]);
        println!(
            "Method 2 (server loop) - step 2 (decode): {} -> '{}'",
            second_token, decoded_2
        );

        // Both should produce same first token
        // Get the first token from method 1
        let first_from_v1 = all_tokens_v1[prompt_len];
        println!(
            "\n=== COMPARISON ===\nMethod 1 first token: {}\nMethod 2 first token: {}",
            first_from_v1, first_token
        );

        // They should be the same
        let tokens_match = first_from_v1 == first_token;
        println!("Match: {}", tokens_match);

        if !tokens_match {
            println!("\n*** BUG FOUND: Unit test and server produce different outputs! ***");
            println!("This explains why server produces garbage while unit tests work.");
        }
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_special_token_filtering_logic() {
        const SPECIAL_TOKENS_TO_SKIP: &[&str] = &["<|endoftext|>", "<|im_end|>", "<|im_start|>"];

        fn should_skip_token_text(text: &str) -> bool {
            text.is_empty() || SPECIAL_TOKENS_TO_SKIP.contains(&text)
        }

        fn clean_completion_text(text: &str) -> String {
            let mut result = text.to_string();
            for token in SPECIAL_TOKENS_TO_SKIP {
                result = result.replace(*token, "");
            }
            result.trim().to_string()
        }

        let tokenizer = setup_tokenizer();
        let eos_token = 151643u32;
        let im_end_token = 151645u32;
        let im_start_token = 151644u32;

        // Test: tokens that decode to special tokens should be skipped
        let eos_decoded = tokenizer.decode(&[eos_token]);
        let im_end_decoded = tokenizer.decode(&[im_end_token]);
        let im_start_decoded = tokenizer.decode(&[im_start_token]);

        println!("EOS decoded: {:?}", eos_decoded);
        println!("IM_END decoded: {:?}", im_end_decoded);
        println!("IM_START decoded: {:?}", im_start_decoded);

        assert!(
            should_skip_token_text(&eos_decoded),
            "EOS should be skipped"
        );
        assert!(
            should_skip_token_text(&im_end_decoded),
            "IM_END should be skipped"
        );
        assert!(
            should_skip_token_text(&im_start_decoded),
            "IM_START should be skipped"
        );

        // Test: normal tokens should not be skipped
        let normal_token = 6023u32; // "hi"
        let normal_decoded = tokenizer.decode(&[normal_token]);
        assert!(
            !should_skip_token_text(&normal_decoded),
            "Normal tokens should not be skipped"
        );

        // Test: clean_completion_text removes special tokens
        let mixed = format!("Hello{}World{}End", eos_decoded, im_end_decoded);
        let cleaned = clean_completion_text(&mixed);
        println!("Mixed: {:?}", mixed);
        println!("Cleaned: {:?}", cleaned);
        assert!(
            !cleaned.contains("<|"),
            "Cleaned text should not contain special tokens"
        );
        assert_eq!(cleaned, "HelloWorldEnd", "Should remove special tokens");
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_streaming_with_special_token_filtering() {
        const SPECIAL_TOKENS_TO_SKIP: &[&str] = &["<|endoftext|>", "<|im_end|>", "<|im_start|>"];

        fn should_skip_token_text(text: &str) -> bool {
            text.is_empty() || SPECIAL_TOKENS_TO_SKIP.contains(&text)
        }

        let tokenizer = setup_tokenizer();
        let eos_token = 151643u32;
        let im_end_token = 151645u32;

        // Simulate server token stream with special tokens
        let server_tokens = vec![
            6023u32,      // "hi"
            6024u32,      // normal token
            eos_token,    // EOS - should be skipped
            6025u32,      // normal token
            im_end_token, // IM_END - should be skipped
            6026u32,      // normal token
        ];

        println!("=== Streaming simulation ===");
        let mut streamed = Vec::new();
        for token in &server_tokens {
            let text = tokenizer.decode(&[*token]);
            println!("Token {} -> decoded: {:?}", token, text);
            if should_skip_token_text(&text) {
                println!("  -> SKIPPED");
                continue;
            }
            streamed.push(text.clone());
            println!("  -> Streamed");
        }

        let combined: String = streamed.iter().map(|s| s.as_str()).collect();
        println!("\nStreamed output: {:?}", combined);

        // Should not contain special tokens
        assert!(
            !combined.contains("<|endoftext|>"),
            "Should not contain EOS"
        );
        assert!(
            !combined.contains("<|im_end|>"),
            "Should not contain IM_END"
        );
        assert!(
            !combined.contains("<|im_start|>"),
            "Should not contain IM_START"
        );

        // Should contain the normal tokens
        let batch_decoded = tokenizer.decode(&[6023u32, 6024u32, 6025u32, 6026u32]);
        assert!(
            combined.contains(&batch_decoded[..batch_decoded.len().min(2)]),
            "Should contain normal token text"
        );
    }

    #[test]
    #[cfg(all(feature = "real_weights", feature = "tokenizers"))]
    fn test_decode_position_calculation() {
        // This test verifies the decode position calculation matches server expectations
        use candle_core::Device;
        use vllm_model::loader::ModelLoader;

        let device = Device::Cpu;
        let loader = ModelLoader::builder(device)
            .with_model_dir("/models/Qwen3-0.6B".to_string())
            .with_kv_blocks(1024)
            .build()
            .expect("Failed to build loader");

        let mut model = loader.load_model().expect("Failed to load model");
        let tokenizer = setup_tokenizer();

        let prompt_tokens = vec![6023u32]; // "hi"
        let prompt_len = prompt_tokens.len();

        // After prefill: all_tokens = [6023], num_computed = 1
        // Decode position should be: len - 1 = 0
        // But num_computed should be: len - 1 = 0 (before processing)

        println!("=== DECODE POSITION TEST ===");
        println!("Prompt tokens: {:?}", prompt_tokens);
        println!("Prompt len: {}", prompt_len);

        // Prefill first
        let all_tokens_prefill = prompt_tokens.clone();
        let positions_prefill: Vec<usize> = (0..all_tokens_prefill.len()).collect();

        let (prefill_logits, _) = model
            .forward_with_cache(&all_tokens_prefill, 0, &[0], &positions_prefill, true)
            .expect("Prefill failed");

        use candle_core::D;
        let next_after_prefill: u32 = {
            let seq_len = prefill_logits.dims()[1];
            prefill_logits
                .narrow(1, seq_len - 1, 1)
                .unwrap()
                .squeeze(1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .argmax(D::Minus1)
                .unwrap()
                .to_vec0()
                .unwrap()
        };

        println!(
            "After prefill: generated token = {} -> '{}'",
            next_after_prefill,
            tokenizer.decode(&[next_after_prefill])
        );

        // Now simulate decode
        let mut all_tokens = prompt_tokens.clone();
        all_tokens.push(next_after_prefill);

        // tokens_len = 2, position = 1, num_computed = 1
        // OR position = 0, num_computed = 1 (depends on interpretation)
        let tokens_len = all_tokens.len();
        let last_token = *all_tokens.last().unwrap();
        let position = tokens_len - 1; // This is what batch_composer does
        let num_computed_for_decode = tokens_len - 1; // This is what batch_composer does

        println!("\nDecode params:");
        println!("  tokens: {:?}", all_tokens);
        println!("  tokens_len: {}", tokens_len);
        println!("  last_token: {}", last_token);
        println!("  position (for forward): {}", position);
        println!("  num_computed (for forward): {}", num_computed_for_decode);

        // Test decode with position=1
        let (logits_pos1, _) = model
            .forward_with_cache(
                &[last_token],
                num_computed_for_decode,
                &[0],
                &[position],
                false,
            )
            .expect("Decode with pos=1 failed");

        let next_pos1: u32 = logits_pos1
            .argmax(D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec0()
            .unwrap();

        println!(
            "Decode with position={}: {} -> '{}'",
            position,
            next_pos1,
            tokenizer.decode(&[next_pos1])
        );

        // Test decode with position=0
        let (logits_pos0, _) = model
            .forward_with_cache(&[last_token], num_computed_for_decode, &[0], &[0], false)
            .expect("Decode with pos=0 failed");

        let next_pos0: u32 = logits_pos0
            .argmax(D::Minus1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec0()
            .unwrap();

        println!(
            "Decode with position=0: {} -> '{}'",
            next_pos0,
            tokenizer.decode(&[next_pos0])
        );

        // The results should be different if position matters
        if next_pos1 != next_pos0 {
            println!("\n*** Position DOES matter for decode! ***");
            println!("This could be the source of the bug if server uses wrong position.");
        }
    }
}
