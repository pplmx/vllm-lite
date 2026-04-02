use vllm_traits::{BatchOutput, SeqId, TokenId};
use vllm_traits::{ModelBackend, Result};

pub struct IncrementModel;

impl ModelBackend for IncrementModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|id| *id as TokenId).collect(),
        })
    }

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|tokens| tokens.iter().map(|_| 0.0).collect())
            .collect())
    }
}

#[derive(Clone)]
pub struct ConstModel {
    pub return_token: TokenId,
}

impl ConstModel {
    pub fn new(return_token: TokenId) -> Self {
        Self { return_token }
    }
}

impl ModelBackend for ConstModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| self.return_token).collect(),
        })
    }

    fn forward_logits(
        &self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment_model() {
        let mut model = IncrementModel;
        let output = model
            .forward(&[1, 2], &[vec![1], vec![2]], &[vec![0], vec![0]], &[vec![0], vec![0]], &[0, 0], &[true, true])
            .unwrap();
        assert_eq!(output.next_tokens, vec![1, 2]);
    }

    #[test]
    fn test_const_model() {
        let mut model = ConstModel::new(42);
        let output = model.forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true]).unwrap();
        assert_eq!(output.next_tokens, vec![42]);
    }
}
