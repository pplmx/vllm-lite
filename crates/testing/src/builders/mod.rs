//! Test builders for common structures.

use vllm_traits::{Batch, BatchPhase, SeqId, TokenId};

#[derive(Debug)]
/// Builder for `Request`. Use `with_*` methods to override defaults, then call `.build()` to produce the final value.
pub struct RequestBuilder {
    seq_id: SeqId,
    tokens: Vec<TokenId>,
    max_tokens: usize,
}

impl RequestBuilder {
    /// Create a new builder for the given sequence ID.
    #[must_use]
    pub const fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            tokens: vec![],
            max_tokens: 16,
        }
    }

    /// Replace the prompt token list for this request.
    #[must_use]
    pub fn with_prompt(mut self, tokens: Vec<TokenId>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set the `max_tokens` generation cap for this request.
    #[must_use]
    pub const fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Consume the builder and produce the final `(seq_id, tokens, max_tokens)` tuple.
    #[must_use]
    pub fn build(self) -> (SeqId, Vec<TokenId>, usize) {
        (self.seq_id, self.tokens, self.max_tokens)
    }
}
#[derive(Debug)]

/// Builder for `Batch`. Use `with_*` methods to override defaults, then call `.build()` to produce the final value.
pub struct BatchBuilder {
    seq_ids: Vec<SeqId>,
    input_tokens: Vec<Vec<TokenId>>,
    positions: Vec<Vec<usize>>,
    kv_block_ids: Vec<Vec<usize>>,
    num_computed_tokens: Vec<usize>,
    is_prefill: Vec<bool>,
}

impl BatchBuilder {
    /// Create an empty builder. Use `add_sequence` to populate it.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            seq_ids: vec![],
            input_tokens: vec![],
            positions: vec![],
            kv_block_ids: vec![],
            num_computed_tokens: vec![],
            is_prefill: vec![],
        }
    }

    /// Add a sequence to the batch with its token list and prefill/decode flag.
    #[must_use]
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

    /// Consume the builder and produce the final [`Batch`] value.
    #[must_use]
    pub fn build(self) -> Batch {
        let total_tokens: usize = self.input_tokens.iter().map(std::vec::Vec::len).sum();
        let max_seq_len = self
            .input_tokens
            .iter()
            .map(std::vec::Vec::len)
            .max()
            .unwrap_or(0);

        // Determine phase based on is_prefill
        let phase = if self.is_prefill.iter().all(|&p| p) {
            BatchPhase::Prefill
        } else if self.is_prefill.iter().all(|&p| !p) {
            BatchPhase::Decode
        } else {
            BatchPhase::Mixed
        };

        let n = self.seq_ids.len();
        Batch {
            seq_ids: self.seq_ids,
            input_tokens: self.input_tokens,
            positions: self.positions,
            kv_block_ids: self.kv_block_ids,
            num_computed_tokens: self.num_computed_tokens,
            is_prefill: self.is_prefill,
            sampling_params: std::iter::repeat_with(vllm_traits::SamplingParams::default)
                .take(n)
                .collect(),
            phase,
            total_tokens,
            max_seq_len,
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
