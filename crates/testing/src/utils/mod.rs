//! Utility functions for testing.

use vllm_traits::{Batch, SeqId, TokenId};

pub fn generate_random_tokens(len: usize) -> Vec<TokenId> {
    (0..len)
        .map(|_| rand::random::<TokenId>() % 32000)
        .collect()
}

pub fn assert_batch_consistency(batch: &Batch) {
    let n = batch.seq_ids.len();
    assert_eq!(batch.input_tokens.len(), n, "input_tokens length mismatch");
    assert_eq!(batch.positions.len(), n, "positions length mismatch");
    assert_eq!(batch.kv_block_ids.len(), n, "kv_block_ids length mismatch");
    assert_eq!(
        batch.num_computed_tokens.len(),
        n,
        "num_computed_tokens length mismatch"
    );
    assert_eq!(batch.is_prefill.len(), n, "is_prefill length mismatch");
}

pub fn create_simple_batch(seq_ids: &[SeqId], token: TokenId) -> Batch {
    let total_tokens: usize = seq_ids.len();
    Batch {
        seq_ids: seq_ids.to_vec(),
        input_tokens: seq_ids.iter().map(|_| vec![token]).collect(),
        positions: seq_ids.iter().map(|_| vec![0]).collect(),
        kv_block_ids: seq_ids.iter().map(|_| vec![0]).collect(),
        num_computed_tokens: vec![0; seq_ids.len()],
        is_prefill: vec![true; seq_ids.len()],
        phase: vllm_traits::BatchPhase::Prefill,
        total_tokens,
        max_seq_len: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_tokens() {
        let tokens = generate_random_tokens(10);
        assert_eq!(tokens.len(), 10);
    }

    #[test]
    fn test_assert_batch_consistency_valid() {
        let batch = create_simple_batch(&[1, 2, 3], 42);
        assert_batch_consistency(&batch);
    }

    #[test]
    fn test_assert_batch_consistency_invalid() {
        let batch = Batch {
            seq_ids: vec![1, 2],
            input_tokens: vec![vec![1]], // mismatched length
            positions: vec![vec![0]],
            kv_block_ids: vec![vec![0]],
            num_computed_tokens: vec![0],
            is_prefill: vec![true],
            phase: vllm_traits::BatchPhase::Prefill,
            total_tokens: 1,
            max_seq_len: 1,
        };
        // This should panic
        let _ = std::panic::catch_unwind(|| assert_batch_consistency(&batch));
    }
}
