use crate::ModelBackend;
use rand::RngExt;
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, SeqId, TokenId};

pub struct FakeModel {
    vocab_size: usize,
}

impl FakeModel {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl ModelBackend for FakeModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<BatchOutput> {
        let mut rng = rand::rng();
        let next_tokens: Vec<TokenId> = seq_ids
            .iter()
            .map(|_| rng.random_range(0..self.vocab_size) as TokenId)
            .collect();

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }
}
