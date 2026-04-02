use crate::types::{BlockId, TokenId};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct BeamSequence {
    pub tokens: Vec<TokenId>,
    pub score: f32,
    pub kv_blocks: Arc<Vec<BlockId>>,
}

impl BeamSequence {
    pub fn new(tokens: Vec<TokenId>, score: f32, kv_blocks: Vec<BlockId>) -> Self {
        Self {
            tokens,
            score,
            kv_blocks: Arc::new(kv_blocks),
        }
    }

    pub fn push(&mut self, token: TokenId, log_prob: f32) {
        self.tokens.push(token);
        self.score += log_prob;
    }
}
