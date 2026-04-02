use crate::types::{BatchOutput, SeqId, TokenId};

#[derive(Debug)]
pub struct ModelError {
    msg: String,
}

impl ModelError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model error: {}", self.msg)
    }
}

impl std::error::Error for ModelError {}

pub type Result<T> = std::result::Result<T, ModelError>;

pub trait ModelBackend: Send + Sync {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<BatchOutput>;

    fn forward_logits(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>>;
}
