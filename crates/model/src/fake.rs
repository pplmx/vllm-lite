use vllm_core::engine::ModelBackend;
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
        let next_tokens: Vec<TokenId> = seq_ids
            .iter()
            .map(|&id| ((id as usize) % self.vocab_size) as TokenId)
            .collect();

        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens,
        })
    }
}
