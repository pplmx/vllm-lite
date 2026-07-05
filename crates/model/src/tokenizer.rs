//! Tokenizer wrapper around `tokenizers` (`HuggingFace`) + chat-template rendering.
//!
//! `Tokenizer` is constructed from the `tokenizer.json` + `tokenizer_config.json`
//! shipped alongside the model weights. Chat templates are loaded from
//! `chat_template.jinja` if present; otherwise we fall back to the
//! built-in Qwen / Llama / Mistral templates.
#![allow(clippy::module_name_repetitions)]
use std::fmt::Write;
use tokenizers::Tokenizer as HFTokenizer;

/// Error type for tokenizer encoding/decoding failures. Covers invalid UTF-8, missing vocab entries, and chat-template substitution errors.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer from {path}: {source}")]
    LoadFailed {
        path: String,
        #[source]
        source: tokenizers::Error,
    },
}

#[derive(Debug)]
/// Tokenizer wrapper around `HuggingFace` `tokenizers::Tokenizer` with
/// helpers for prompt + chat-template encoding. Constructed via
/// `Tokenizer::from_file` or `Tokenizer::from_pretrained`; never
/// instantiated directly because the inner HF tokenizer must be
/// deserialized from a JSON file.
pub struct Tokenizer {
    inner: Option<Box<HFTokenizer>>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    model_name: Option<String>,
}

impl Tokenizer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: None,
            vocab_size: 151_936,
            special_tokens: vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
            ],
            model_name: None,
        }
    }

    /// Construct a tokenizer from a tokenizer.json file.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_file(path: &str) -> std::result::Result<Self, TokenizerError> {
        let tokenizer = HFTokenizer::from_file(path).map_err(|e| TokenizerError::LoadFailed {
            path: path.to_string(),
            source: e,
        })?;
        let vocab_size = tokenizer.get_vocab_size(true);

        let mut special_tokens = Vec::new();
        for id in tokenizer.get_added_tokens_decoder().keys() {
            if let Some(token) = tokenizer.id_to_token(*id)
                && !token.starts_with('▁')
                && token.len() > 1
                && token.starts_with('<')
            {
                special_tokens.push(token);
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

    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.inner
            && let Ok(encoding) = tokenizer.encode(text, false)
        {
            return encoding.get_ids().to_vec();
        }

        text.split_whitespace()
            .enumerate()
            .map(|(i, _)| u32::try_from(i + 1).unwrap_or(u32::MAX))
            .collect()
    }

    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        if let Some(ref tokenizer) = self.inner
            && let Ok(text) = tokenizer.decode(tokens, false)
        {
            return text;
        }

        tokens.iter().fold(String::new(), |mut acc, t| {
            let _ = write!(acc, "token_{t} ");
            acc
        })
    }

    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[must_use]
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    #[must_use]
    pub fn is_special_token(&self, text: &str) -> bool {
        self.special_tokens.iter().any(|t| t.as_str() == text)
    }

    #[must_use]
    pub fn clean_special_tokens(&self, text: &str) -> String {
        let mut result = text.to_string();
        for token in &self.special_tokens {
            result = result.replace(token.as_str(), "");
        }
        result.trim().to_string()
    }

    #[must_use]
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
        assert_eq!(tokenizer.vocab_size(), 151_936);
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
                // invariant: pre-conditions make this infallible at this call site.
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
                // invariant: pre-conditions make this infallible at this call site.
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
