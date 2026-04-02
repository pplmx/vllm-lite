use serde::{Deserialize, Serialize};

pub const BLOCK_SIZE: usize = 16;

pub type TokenId = u32;
pub type SeqId = u64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}
