use serde::{Deserialize, Serialize};

pub const BLOCK_SIZE: usize = 16;
pub type BlockId = usize;
pub type TokenId = u32;
pub type SeqId = u64;

/// Batch phase
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchPhase {
    Prefill,
    Decode,
    Mixed,
}

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
    pub fn empty() -> Self {
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
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    /// Get batch length
    pub fn len(&self) -> usize {
        self.seq_ids.len()
    }

    pub fn has_prefill(&self) -> bool {
        self.is_prefill.iter().any(|&p| p)
    }

    pub fn has_decode(&self) -> bool {
        self.is_prefill.iter().any(|&p| !p)
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchOutput {
    pub seq_ids: Vec<SeqId>,
    pub next_tokens: Vec<TokenId>,
}

#[derive(Debug, Clone)]
pub enum TensorParallelError {
    InvalidWorldSize,
    InvalidRank,
    DeviceMismatch,
    InputSizeMismatch,
    AllReduceFailed(String),
    CudaError(String),
}

impl std::fmt::Display for TensorParallelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWorldSize => write!(f, "World size must be > 0"),
            Self::InvalidRank => write!(f, "Rank must be < world size"),
            Self::DeviceMismatch => write!(f, "Number of device IDs must match world size"),
            Self::InputSizeMismatch => {
                write!(f, "Input size does not match expected size per rank")
            }
            Self::AllReduceFailed(msg) => write!(f, "All-reduce failed: {}", msg),
            Self::CudaError(msg) => write!(f, "CUDA error: {}", msg),
        }
    }
}

impl std::error::Error for TensorParallelError {}
