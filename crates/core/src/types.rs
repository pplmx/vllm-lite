use std::sync::Arc;
use tokio::sync::mpsc;
pub use vllm_traits::{Batch, BatchOutput, BlockId, SeqId, TokenId};

pub use crate::kv_cache::BLOCK_SIZE;
use crate::scheduler::cuda_graph::SchedulerCudaGraphConfig;

/// Configuration for adaptive speculative decoding
#[derive(Clone, Debug)]
pub struct AdaptiveDraftConfig {
    /// Minimum number of draft tokens
    pub min_draft_tokens: usize,
    /// Maximum number of draft tokens
    pub max_draft_tokens: usize,
    /// Target acceptance rate (0.0-1.0)
    pub target_acceptance_rate: f32,
    /// Window size for accuracy tracking
    pub accuracy_window_size: usize,
    /// Adjustment step size
    pub adjustment_step: usize,
    /// Cooldown steps between adjustments
    pub cooldown_steps: usize,
    /// EWMA smoothing factor (0.0-1.0). Higher = more responsive to recent changes.
    pub ewma_alpha: f32,
    /// Deadband threshold for hysteresis. Only adjusts when |rate - target| > threshold.
    pub deadband_threshold: f32,
}

impl Default for AdaptiveDraftConfig {
    fn default() -> Self {
        Self {
            min_draft_tokens: 2,
            max_draft_tokens: 8,
            target_acceptance_rate: 0.7,
            accuracy_window_size: 20,
            adjustment_step: 1,
            cooldown_steps: 5,
            ewma_alpha: 0.1,
            deadband_threshold: 0.05,
        }
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Prefill,
    Decode,
}

/// Configuration for sequence packing optimization
#[derive(Clone, Debug)]
pub struct SequencePackingConfig {
    /// Enable sequence packing optimization
    pub enabled: bool,
    /// Target batch size for packing
    pub target_batch_size: usize,
    /// Maximum batch size (hard limit)
    pub max_batch_size: usize,
    /// Length similarity threshold (0.0-1.0)
    pub similarity_threshold: f32,
}

impl Default for SequencePackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_batch_size: 32,
            max_batch_size: 256,
            similarity_threshold: 0.2,
        }
    }
}

impl SequencePackingConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let enabled = std::env::var("VLLM_SEQ_PACKING_ENABLED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(true);
        let target_batch_size = std::env::var("VLLM_SEQ_PACKING_TARGET_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let max_batch_size = std::env::var("VLLM_SEQ_PACKING_MAX_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let similarity_threshold = std::env::var("VLLM_SEQ_PACKING_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.2);
        Self {
            enabled,
            target_batch_size,
            max_batch_size,
            similarity_threshold,
        }
    }
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
    /// CUDA Graph configuration
    pub cuda_graph: SchedulerCudaGraphConfig,
    /// Sequence packing configuration
    pub packing: SequencePackingConfig,
}

impl SchedulerConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        max_consecutive_decode: u32,
        enable_pd_separation: bool,
        prefill_chunk_size: usize,
        decode_preference_ratio: f32,
        enable_priority_scheduling: bool,
        enable_dynamic_batching: bool,
        min_batch_size: usize,
        max_batch_size: usize,
        packing: SequencePackingConfig,
    ) -> Self {
        assert!(max_num_seqs > 0, "max_num_seqs must be > 0");
        assert!(
            max_num_batched_tokens > 0,
            "max_num_batched_tokens must be > 0"
        );
        assert!(prefill_chunk_size > 0, "prefill_chunk_size must be > 0");
        assert!(
            (0.0..=1.0).contains(&decode_preference_ratio),
            "decode_preference_ratio must be between 0.0 and 1.0"
        );
        assert!(min_batch_size > 0, "min_batch_size must be > 0");
        assert!(
            max_batch_size >= min_batch_size,
            "max_batch_size must be >= min_batch_size"
        );
        assert!(
            max_num_batched_tokens >= max_batch_size,
            "max_num_batched_tokens must be >= max_batch_size"
        );

        Self {
            max_num_seqs,
            max_num_batched_tokens,
            max_consecutive_decode,
            enable_pd_separation,
            prefill_chunk_size,
            decode_preference_ratio,
            enable_priority_scheduling,
            enable_dynamic_batching,
            min_batch_size,
            max_batch_size,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing,
        }
    }
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
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing: SequencePackingConfig::default(),
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
