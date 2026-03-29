use super::request::TokenId;

pub type BlockId = usize;

pub struct Sequence {
    pub id: u64,
    pub tokens: Vec<TokenId>,
    pub kv_blocks: Vec<BlockId>,
    pub status: Status,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Status {
    Prefill,
    Decoding,
    Finished,
}
