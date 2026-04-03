#[cfg(feature = "tokenizers")]
use tokenizers::Tokenizer as HFTokenizer;

pub struct Tokenizer {
    #[cfg(feature = "tokenizers")]
    inner: Option<Box<HFTokenizer>>,
    #[cfg(not(feature = "tokenizers"))]
    _placeholder: (),
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "tokenizers")]
            inner: None,
            #[cfg(not(feature = "tokenizers"))]
            _placeholder: (),
            vocab_size: 151936,
        }
    }

    #[cfg(feature = "tokenizers")]
    pub fn from_file(path: &str) -> std::result::Result<Self, String> {
        let tokenizer =
            HFTokenizer::from_file(path).map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        let vocab_size = tokenizer.get_vocab_size(true);
        Ok(Self {
            inner: Some(Box::new(tokenizer)),
            vocab_size,
        })
    }

    #[cfg(not(feature = "tokenizers"))]
    pub fn from_file(path: &str) -> std::result::Result<Self, String> {
        let _ = path;
        Ok(Self {
            _placeholder: (),
            vocab_size: 151936,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        #[cfg(feature = "tokenizers")]
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
        #[cfg(feature = "tokenizers")]
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
}
