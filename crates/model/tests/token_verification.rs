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
