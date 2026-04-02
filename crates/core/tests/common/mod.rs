use vllm_core::engine::ModelBackend;
use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, SeqId, TokenId};

pub struct IncrementModel;

impl ModelBackend for IncrementModel {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
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
        &self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
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
        let model = IncrementModel;
        let output = model
            .forward(&[1, 2], &[vec![1], vec![2]], &[vec![0], vec![0]])
            .unwrap();
        assert_eq!(output.next_tokens, vec![1, 2]);
    }

    #[test]
    fn test_const_model() {
        let model = ConstModel::new(42);
        let output = model.forward(&[1], &[vec![1]], &[vec![0]]).unwrap();
        assert_eq!(output.next_tokens, vec![42]);
    }
}
