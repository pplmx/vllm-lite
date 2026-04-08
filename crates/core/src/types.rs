use std::sync::Arc;
use tokio::sync::mpsc;
pub use vllm_traits::{Batch, BatchOutput, BlockId, SeqId, TokenId};

pub use crate::kv_cache::BLOCK_SIZE;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Priority(pub u8);

#[derive(Clone, Debug)]
pub struct Request {
    pub id: SeqId,
    pub prompt: Vec<TokenId>,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub priority: Priority,
}

impl Request {
    pub fn new(id: SeqId, prompt: Vec<TokenId>, max_tokens: usize) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            sampling_params: SamplingParams::default(),
            priority: Priority::default(),
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub beam_width: usize,
    pub length_penalty: f32,
    pub max_retries: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            beam_width: 1,
            length_penalty: 0.6,
            max_retries: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub id: SeqId,
    pub tokens: Vec<TokenId>,
    pub kv_blocks: Arc<Vec<BlockId>>,
    pub num_computed_tokens: usize,
    pub prompt_len: usize,
    pub status: Status,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub consecutive_decode_rounds: u32,
    pub priority: Priority,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// Configuration for the request scheduler.
///
/// Controls batching behavior, prefill/decode separation, and priority handling.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Maximum number of sequences that can be scheduled in a single batch.
    pub max_num_seqs: usize,
    /// Maximum number of tokens (including prompt and generated) in a batch.
    pub max_num_batched_tokens: usize,
    /// Maximum consecutive decode iterations before forcing a prefill.
    pub max_consecutive_decode: u32,
    /// Enable separation of prefill and decode phases into different batches.
    pub enable_pd_separation: bool,
    /// Maximum number of prompt tokens to process in a single prefill chunk.
    pub prefill_chunk_size: usize,
    /// Ratio of decode-to-prefill tokens when batching mixed phases (0.0-1.0).
    pub decode_preference_ratio: f32,
    /// Enable priority-based scheduling (higher priority requests first).
    pub enable_priority_scheduling: bool,
    /// Enable dynamic batching (grouping similar requests automatically).
    pub enable_dynamic_batching: bool,
    /// Minimum batch size for dynamic batching.
    pub min_batch_size: usize,
    /// Maximum batch size for dynamic batching.
    pub max_batch_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_consecutive_decode: 10,
            enable_pd_separation: true,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: true,
            min_batch_size: 1,
            max_batch_size: 256,
        }
    }
}

use crate::metrics::MetricsSnapshot;

pub enum EngineMessage {
    AddRequest {
        request: Request,
        response_tx: mpsc::Sender<TokenId>,
    },
    GetMetrics {
        response_tx: mpsc::UnboundedSender<MetricsSnapshot>,
    },
    GetEmbeddings {
        input_tokens: Vec<Vec<TokenId>>,
        response_tx: mpsc::UnboundedSender<Vec<Vec<f32>>>,
    },
    Shutdown,
}
