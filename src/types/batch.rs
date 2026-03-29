use crate::types::sequence::BlockId;

pub struct Batch {
    pub input_tokens: Vec<u32>,
    pub seq_ids: Vec<u64>,
    pub kv_map: Vec<Vec<BlockId>>,
}
