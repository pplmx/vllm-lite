use tokenizers::Tokenizer as HFTokenizer;

pub struct Tokenizer {
    inner: Option<Box<HFTokenizer>>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    model_name: Option<String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            inner: None,
            vocab_size: 151936,
            special_tokens: vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
            ],
            model_name: None,
        }
    }

    pub fn from_file(path: &str) -> std::result::Result<Self, String> {
        let tokenizer =
            HFTokenizer::from_file(path).map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        let vocab_size = tokenizer.get_vocab_size(true);

        let mut special_tokens = Vec::new();
        for id in tokenizer.get_added_tokens_decoder().keys() {
            if let Some(token) = tokenizer.id_to_token(*id) {
                if !token.starts_with('▁') && token.len() > 1 && token.starts_with('<') {
                    special_tokens.push(token);
                }
            }
        }

        if special_tokens.is_empty() {
            special_tokens = vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
            ];
        }

        Ok(Self {
            inner: Some(Box::new(tokenizer)),
            vocab_size,
            special_tokens,
            model_name: Some("Qwen3.5-0.8B".to_string()),
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.inner {
            if let Ok(encoding) = tokenizer.encode(text, false) {
                return encoding.get_ids().to_vec();
            }
        }

        text.split_whitespace()
            .enumerate()
            .map(|(i, _)| (i + 1) as u32)
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        if let Some(ref tokenizer) = self.inner {
            if let Ok(text) = tokenizer.decode(tokens, false) {
                return text;
            }
        }

        tokens.iter().map(|t| format!("token_{} ", t)).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    pub fn is_special_token(&self, text: &str) -> bool {
        self.special_tokens.iter().any(|t| t.as_str() == text)
    }

    pub fn clean_special_tokens(&self, text: &str) -> String {
        let mut result = text.to_string();
        for token in &self.special_tokens {
            result = result.replace(token.as_str(), "");
        }
        result.trim().to_string()
    }

    pub fn model_name(&self) -> Option<String> {
        self.model_name.clone()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = Tokenizer::new();
        let _ = tokenizer.encode("test");
    }

    #[test]
    fn test_tokenizer_encode_simple() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("hello world");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[1], 2);
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 151936);
    }

    #[test]
    fn test_tokenizer_encode_empty() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("");

        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_single_word() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("hello");

        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_tokenizer_decode_single_token() {
        let tokenizer = Tokenizer::new();
        let text = tokenizer.decode(&[1]);

        assert!(!text.is_empty());
    }

    #[test]
    fn test_tokenizer_qwen3_hi_token() {
        use std::path::PathBuf;

        let model_path = PathBuf::from("/models/Qwen3-0.6B");
        let tokenizer_path = model_path.join("tokenizer.json");

        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .expect("Failed to load tokenizer");

            let tokens = tokenizer.encode("hi");
            assert_eq!(tokens.len(), 1, "hi should be a single token");
            assert_eq!(tokens[0], 6023, "hi should be token 6023");

            let decoded = tokenizer.decode(&tokens);
            assert!(decoded.contains("hi"), "Decoded text should contain 'hi'");
        }
    }

    #[test]
    fn test_tokenizer_qwen3_chat_prompt() {
        use std::path::PathBuf;

        let model_path = PathBuf::from("/models/Qwen3-0.6B");
        let tokenizer_path = model_path.join("tokenizer.json");

        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .expect("Failed to load tokenizer");

            let im_start = "<|im_start|>";
            let im_end = "<|im_end|>";
            let prompt = format!(
                "{}user\nhi{}{}\n{}assistant\n",
                im_start, im_end, "\n", im_start
            );

            let tokens = tokenizer.encode(&prompt);
            assert!(!tokens.is_empty(), "Chat prompt should produce tokens");
            assert!(
                tokens.len() < 100,
                "Chat prompt should be reasonable length"
            );

            let decoded = tokenizer.decode(&tokens);
            assert!(!decoded.is_empty(), "Should be able to decode tokens");
        }
    }
}
