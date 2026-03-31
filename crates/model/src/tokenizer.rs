#[cfg(feature = "real_weights")]
use tiktoken::{Cl100KBase, Tokenizer};

#[cfg(feature = "real_weights")]
pub struct Tokenizer {
    inner: Tokenizer<Cl100KBase>,
}

#[cfg(feature = "real_weights")]
impl Tokenizer {
    pub fn new() -> Self {
        Self {
            inner: Cl100KBase.into(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text).unwrap_or_default()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens).unwrap_or_default()
    }
}

#[cfg(feature = "real_weights")]
impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "real_weights"))]
pub struct Tokenizer;

#[cfg(not(feature = "real_weights"))]
impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, _)| (i + 1) as u32)
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|t| format!("token_{} ", t)).collect()
    }
}

#[cfg(not(feature = "real_weights"))]
impl Default for Tokenizer {
    fn default() -> Self {
        Self
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
    fn test_tokenizer_encode_multiple_words() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("the quick brown fox");

        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_tokenizer_encode_whitespace_handling() {
        let tokenizer = Tokenizer::new();

        let tokens1 = tokenizer.encode("hello world");
        let tokens2 = tokenizer.encode("hello  world");

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_tokenizer_decode_empty() {
        let tokenizer = Tokenizer::new();
        let text = tokenizer.decode(&[]);

        assert!(text.is_empty());
    }

    #[test]
    fn test_tokenizer_decode_single_token() {
        let tokenizer = Tokenizer::new();
        let text = tokenizer.decode(&[1]);

        assert!(!text.is_empty());
    }

    #[test]
    fn test_tokenizer_decode_multiple_tokens() {
        let tokenizer = Tokenizer::new();
        let text = tokenizer.decode(&[1, 2, 3, 4, 5]);

        assert!(text.contains("token_1"));
        assert!(text.contains("token_5"));
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let tokenizer = Tokenizer::new();
        let original = "hello world";

        let tokens = tokenizer.encode(original);
        let decoded = tokenizer.decode(&tokens);

        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_unicode() {
        let tokenizer = Tokenizer::new();

        let tokens = tokenizer.encode("你好世界");
        assert!(!tokens.is_empty());

        let tokens2 = tokenizer.encode("🎉🎊🎁");
        assert!(!tokens2.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_leading_trailing_spaces() {
        let tokenizer = Tokenizer::new();

        let tokens1 = tokenizer.encode("hello");
        let tokens2 = tokenizer.encode("  hello  ");

        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_tokenizer_consistency() {
        let tokenizer = Tokenizer::new();

        let tokens1 = tokenizer.encode("hello world test");
        let tokens2 = tokenizer.encode("hello world test");

        assert_eq!(tokens1, tokens2);
    }
}
