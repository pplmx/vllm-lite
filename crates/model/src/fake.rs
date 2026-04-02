use vllm_traits::{BatchOutput, SeqId, TokenId};
use vllm_traits::{ModelBackend, Result};

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

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        let vocab_size = 32000;
        Ok(input_tokens
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .map(|_| rand::random::<f32>() * vocab_size as f32)
                    .collect()
            })
            .collect())
    }
}
