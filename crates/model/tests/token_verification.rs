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
}
