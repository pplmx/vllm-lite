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
