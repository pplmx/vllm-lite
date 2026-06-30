use serde::{Deserialize, Serialize};

/// `BLOCK_SIZE`: block size constant.
pub const BLOCK_SIZE: usize = 16;
/// `BlockId`: block id.
pub type BlockId = usize;
/// `TokenId`: token id.
pub type TokenId = u32;
/// `SeqId`: seq id.
pub type SeqId = u64;

/// Batch phase
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchPhase {
    Prefill,
    Decode,
    Mixed,
}

/// Batch: batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    // KV cache information
    pub kv_block_ids: Vec<Vec<BlockId>>,
    pub num_computed_tokens: Vec<usize>,
    pub is_prefill: Vec<bool>,
    // New fields
    pub phase: BatchPhase,
    pub total_tokens: usize,
    pub max_seq_len: usize,
}

impl Batch {
    /// Create an empty batch
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            seq_ids: Vec::new(),
            input_tokens: Vec::new(),
            positions: Vec::new(),
            kv_block_ids: Vec::new(),
            num_computed_tokens: Vec::new(),
            is_prefill: Vec::new(),
            phase: BatchPhase::Mixed,
            total_tokens: 0,
            max_seq_len: 0,
        }
    }

    /// Check if batch is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    /// Get batch length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.seq_ids.len()
    }

    #[must_use]
    pub fn has_prefill(&self) -> bool {
        self.is_prefill.iter().any(|&p| p)
    }

    #[must_use]
    pub fn has_decode(&self) -> bool {
        self.is_prefill.iter().any(|&p| !p)
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::empty()
    }
}

/// `BatchOutput`: batch output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}

/// `TensorParallelError`: tensor parallel error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorParallelError {
    #[error("World size must be > 0")]
    InvalidWorldSize,
    #[error("Rank must be < world size")]
    InvalidRank,
    #[error("Number of device IDs must match world size")]
    DeviceMismatch,
    #[error("Input size does not match expected size per rank")]
    InputSizeMismatch,
    #[error("All-reduce failed: {0}")]
    AllReduceFailed(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
}
