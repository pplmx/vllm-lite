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
            let decoded = tokenizer.decode(&[*token as u32]);

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
                decoded.trim().len() >= 1,
                "Token {} decoded to empty/whitespace: {:?}",
                token,
                decoded
            );
        }
    }
}
