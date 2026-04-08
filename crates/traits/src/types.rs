use serde::{Deserialize, Serialize};

pub const BLOCK_SIZE: usize = 16;
pub type BlockId = usize;
pub type TokenId = u32;
pub type SeqId = u64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub seq_ids: Vec<SeqId>,
    pub input_tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    // KV cache information
    pub kv_block_ids: Vec<Vec<usize>>,
    pub num_computed_tokens: Vec<usize>,
    pub is_prefill: Vec<bool>,
}

impl Batch {
    pub fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    pub fn has_prefill(&self) -> bool {
        self.is_prefill.iter().any(|&p| p)
    }

    pub fn has_decode(&self) -> bool {
        self.is_prefill.iter().any(|&p| !p)
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
