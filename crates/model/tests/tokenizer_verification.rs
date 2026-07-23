//! Qwen3 tokenizer decode and roundtrip tests.

mod support;

#[cfg(test)]
mod tests {
    use super::support;

    #[test]

    fn test_tokenizer_decode_model_output() {
        let tokenizer = support::tokenizer::qwen3_tokenizer();

        let model_output_tokens = vec![13539u32, 47421u32, 60290u32];

        for token in &model_output_tokens {
            let decoded = tokenizer.decode(&[(*token)]);

            println!("Token {token} decodes to: {decoded:?}");

            assert!(
                !decoded.contains('\u{FFFD}'),
                "Token {token} should not decode to replacement char, got: {decoded:?}"
            );

            assert!(
                support::tokenizer::is_printable_text(&decoded),
                "Token {token} should decode to printable text, got: {decoded:?}"
            );

            assert!(
                !decoded.trim().is_empty(),
                "Token {token} decoded to empty/whitespace: {decoded:?}"
            );
        }
    }

    #[test]

    fn test_tokenizer_decode_top_k_tokens() {
        let tokenizer = support::tokenizer::qwen3_tokenizer();

        let sample_ranges = vec![(0, 1000), (10_000, 11_000), (150_000, 151_000)];

        let mut all_samples = Vec::new();
        for (start, end) in sample_ranges {
            for token in start..end {
                all_samples.push(u32::try_from(token).expect("bounded sample range"));
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

        // invariant: fail_count and all_samples.len() are bounded by test sample sizes
        // (max 151_000), so f32 precision loss is acceptable for fail-rate computation.
        #[allow(clippy::cast_precision_loss)]
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

    fn test_tokenizer_roundtrip_vocab() {
        let tokenizer = support::tokenizer::qwen3_tokenizer();

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
                failed.push(((*text).to_string(), tokens, decoded.clone()));
            }
        }

        if !failed.is_empty() {
            println!("Roundtrip failures:");
            for (orig, tokens, decoded) in &failed {
                println!("  '{orig}' -> {tokens:?} -> '{decoded}'");
            }
        }

        // invariant: both lengths are bounded by the small test_strings fixture (8 entries).
        #[allow(clippy::cast_precision_loss)]
        let fail_rate = failed.len() as f32 / test_strings.len() as f32;
        assert!(
            fail_rate < 0.3,
            "{}/{} roundtrip failed",
            failed.len(),
            test_strings.len()
        );
    }

    #[test]

    fn test_qwen3_special_tokens() {
        let tokenizer = support::tokenizer::qwen3_tokenizer();

        let eos_token = 151_643_u32;
        let bos_token = 151_645_u32;

        let eos_decoded = tokenizer.decode(&[eos_token]);
        println!("EOS token {eos_token} decodes to: {eos_decoded:?}");

        let bos_decoded = tokenizer.decode(&[bos_token]);
        println!("BOS token {bos_token} decodes to: {bos_decoded:?}");

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

    fn test_tokenizer_decode_out_of_range_tokens() {
        let tokenizer = support::tokenizer::qwen3_tokenizer();

        let vocab_size = 151_936;
        let out_of_range_tokens = vec![
            vocab_size,       // 151936 - exactly at vocab size
            vocab_size + 1,   // 151937
            vocab_size + 100, // 152036
            200_000,          // Way out of range
            1_000_000,        // Very far out of range
        ];

        let mut replacement_char_count = 0;
        for token in &out_of_range_tokens {
            let decoded = tokenizer.decode(&[*token]);
            println!("Out-of-range token {token} decodes to: {decoded:?}");
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

    fn test_qwen3_eos_handling_in_decode() {
        let tokenizer = support::tokenizer::qwen3_tokenizer();

        let eos_token = 151_643_u32;

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
}
