pub mod fake;

use vllm_core::error::Result;
use vllm_core::types::{BatchOutput, SeqId, TokenId};

pub trait ModelBackend: Send {
    fn forward(
        &self,
        seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        positions: &[Vec<usize>],
    ) -> Result<BatchOutput>;
}
