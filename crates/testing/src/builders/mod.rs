//! Test builders for common structures.

use vllm_traits::{Batch, SeqId, TokenId};

pub struct RequestBuilder {
    seq_id: SeqId,
    tokens: Vec<TokenId>,
    max_tokens: usize,
}

impl RequestBuilder {
    pub fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            tokens: vec![],
            max_tokens: 16,
        }
    }

    pub fn with_prompt(mut self, tokens: Vec<TokenId>) -> Self {
        self.tokens = tokens;
        self
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    pub fn build(self) -> (SeqId, Vec<TokenId>, usize) {
        (self.seq_id, self.tokens, self.max_tokens)
    }
}

pub struct BatchBuilder {
    seq_ids: Vec<SeqId>,
    input_tokens: Vec<Vec<TokenId>>,
    positions: Vec<Vec<usize>>,
    kv_block_ids: Vec<Vec<usize>>,
    num_computed_tokens: Vec<usize>,
    is_prefill: Vec<bool>,
}

impl BatchBuilder {
    pub fn new() -> Self {
        Self {
            seq_ids: vec![],
            input_tokens: vec![],
            positions: vec![],
            kv_block_ids: vec![],
            num_computed_tokens: vec![],
            is_prefill: vec![],
        }
    }

    pub fn add_sequence(mut self, seq_id: SeqId, tokens: Vec<TokenId>, prefill: bool) -> Self {
        let pos = tokens.len();
        self.seq_ids.push(seq_id);
        self.input_tokens.push(tokens);
        self.positions.push((0..pos).collect());
        self.kv_block_ids.push(vec![0]);
        self.num_computed_tokens.push(0);
        self.is_prefill.push(prefill);
        self
    }

    pub fn build(self) -> Batch {
        Batch {
            seq_ids: self.seq_ids,
            input_tokens: self.input_tokens,
            positions: self.positions,
            kv_block_ids: self.kv_block_ids,
            num_computed_tokens: self.num_computed_tokens,
            is_prefill: self.is_prefill,
        }
    }
}

impl Default for BatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_builder() {
        let (seq_id, tokens, max_tokens) = RequestBuilder::new(1)
            .with_prompt(vec![1, 2, 3])
            .with_max_tokens(10)
            .build();

        assert_eq!(seq_id, 1);
        assert_eq!(tokens, vec![1, 2, 3]);
        assert_eq!(max_tokens, 10);
    }

    #[test]
    fn test_batch_builder() {
        let batch = BatchBuilder::new()
            .add_sequence(1, vec![1, 2], true)
            .add_sequence(2, vec![3], false)
            .build();

        assert_eq!(batch.seq_ids.len(), 2);
        assert_eq!(batch.is_prefill, vec![true, false]);
    }
}
