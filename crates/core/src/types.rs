//! types: shared type definitions.

use std::sync::Arc;
use tokio::sync::mpsc;
pub use vllm_traits::{Batch, BatchOutput, BlockId, SeqId, TokenId};

pub use crate::kv_cache::BLOCK_SIZE;
use crate::scheduler::cuda_graph::SchedulerCudaGraphConfig;

pub use crate::speculative::DraftId;

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

impl AdaptiveDraftConfig {
    /// builder: construct via builder for documented field ergonomics.
    pub fn builder() -> AdaptiveDraftConfigBuilder {
        AdaptiveDraftConfigBuilder::default()
    }
}

/// Builder for [`AdaptiveDraftConfig`].
#[derive(Debug, Clone, Default)]
pub struct AdaptiveDraftConfigBuilder {
    inner: AdaptiveDraftConfig,
}

impl AdaptiveDraftConfigBuilder {
    /// with_min_draft_tokens: with min draft tokens.
    pub fn with_min_draft_tokens(mut self, v: usize) -> Self {
        self.inner.min_draft_tokens = v;
        self
    }
    /// with_max_draft_tokens: with max draft tokens.
    pub fn with_max_draft_tokens(mut self, v: usize) -> Self {
        self.inner.max_draft_tokens = v;
        self
    }
    /// with_target_acceptance_rate: with target acceptance rate.
    pub fn with_target_acceptance_rate(mut self, v: f32) -> Self {
        self.inner.target_acceptance_rate = v;
        self
    }
    /// with_accuracy_window_size: with accuracy window size.
    pub fn with_accuracy_window_size(mut self, v: usize) -> Self {
        self.inner.accuracy_window_size = v;
        self
    }
    /// with_adjustment_step: with adjustment step.
    pub fn with_adjustment_step(mut self, v: usize) -> Self {
        self.inner.adjustment_step = v;
        self
    }
    /// with_cooldown_steps: with cooldown steps.
    pub fn with_cooldown_steps(mut self, v: usize) -> Self {
        self.inner.cooldown_steps = v;
        self
    }
    /// with_ewma_alpha: with ewma alpha.
    pub fn with_ewma_alpha(mut self, v: f32) -> Self {
        self.inner.ewma_alpha = v;
        self
    }
    /// with_deadband_threshold: with deadband threshold.
    pub fn with_deadband_threshold(mut self, v: f32) -> Self {
        self.inner.deadband_threshold = v;
        self
    }
    /// build: build the [`AdaptiveDraftConfig`].
    pub fn build(self) -> AdaptiveDraftConfig {
        self.inner
    }
}

/// Priority: priority.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Priority(pub u8);

/// Request: request.
#[derive(Clone, Debug)]
pub struct Request {
    pub id: SeqId,
    pub prompt: Vec<TokenId>,
    pub max_tokens: usize,
    pub sampling_params: SamplingParams,
    pub priority: Priority,
    /// Optional external draft model to use for speculative decoding this
    /// request (v18.0 RTE-01).
    ///
    /// - `None` → no external draft; engine uses self-spec (if enabled) or
    ///   pure target decode.
    /// - `Some(id)` → engine resolves `id` against the `DraftModelRegistry`.
    ///   If the draft cannot be loaded, the engine silently falls back to
    ///   self-spec (FALL-01). If the draft errors at runtime, the request
    ///   degrades to non-spec decode for the remainder of its lifetime
    ///   (FALL-02).
    pub draft_model_id: Option<DraftId>,
}

impl Request {
    /// new: new.
    pub fn new(id: SeqId, prompt: Vec<TokenId>, max_tokens: usize) -> Self {
        Self {
            id,
            prompt,
            max_tokens,
            sampling_params: SamplingParams::default(),
            priority: Priority::default(),
            draft_model_id: None,
        }
    }

    /// with_priority: with priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Bind this request to a specific external draft model. The engine will
    /// resolve `id` against the registry at step time.
    pub fn with_draft_model(mut self, id: impl Into<DraftId>) -> Self {
        self.draft_model_id = Some(id.into());
        self
    }
}

/// SamplingParams: sampling params.
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

impl SamplingParams {
    /// builder: construct via builder for documented field ergonomics.
    pub fn builder() -> SamplingParamsBuilder {
        SamplingParamsBuilder::default()
    }
}

/// Builder for [`SamplingParams`].
#[derive(Debug, Clone, Default)]
pub struct SamplingParamsBuilder {
    inner: SamplingParams,
}

impl SamplingParamsBuilder {
    /// with_temperature: with temperature.
    pub fn with_temperature(mut self, v: f32) -> Self {
        self.inner.temperature = v;
        self
    }
    /// with_top_k: with top k.
    pub fn with_top_k(mut self, v: usize) -> Self {
        self.inner.top_k = v;
        self
    }
    /// with_top_p: with top p.
    pub fn with_top_p(mut self, v: f32) -> Self {
        self.inner.top_p = v;
        self
    }
    /// with_repeat_penalty: with repeat penalty.
    pub fn with_repeat_penalty(mut self, v: f32) -> Self {
        self.inner.repeat_penalty = v;
        self
    }
    /// with_beam_width: with beam width.
    pub fn with_beam_width(mut self, v: usize) -> Self {
        self.inner.beam_width = v;
        self
    }
    /// with_length_penalty: with length penalty.
    pub fn with_length_penalty(mut self, v: f32) -> Self {
        self.inner.length_penalty = v;
        self
    }
    /// with_max_retries: with max retries.
    pub fn with_max_retries(mut self, v: u32) -> Self {
        self.inner.max_retries = v;
        self
    }
    /// build: build the [`SamplingParams`].
    pub fn build(self) -> SamplingParams {
        self.inner
    }
}

/// Sequence: sequence.
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
    /// Set to true when the draft model errored at runtime (v18.0 FALL-02).
    /// While true, the engine routes this sequence through non-spec decode
    /// (no draft attempts). Sticky for the lifetime of the sequence.
    pub degraded_draft: bool,
    /// The external draft model this sequence is bound to (v18.0 RTE-01/02).
    /// `None` means no external draft — engine uses self-spec or non-spec.
    /// Resolved against the `DraftModelRegistry` at step time.
    pub draft_model_id: Option<DraftId>,
}

/// Status: status status.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// Phase: phase phase.
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
    /// builder: construct via builder for documented field ergonomics.
    pub fn builder() -> SequencePackingConfigBuilder {
        SequencePackingConfigBuilder::default()
    }
}

/// Builder for [`SequencePackingConfig`].
#[derive(Debug, Clone, Default)]
pub struct SequencePackingConfigBuilder {
    inner: SequencePackingConfig,
}

impl SequencePackingConfigBuilder {
    /// with_enabled: with enabled.
    pub fn with_enabled(mut self, v: bool) -> Self {
        self.inner.enabled = v;
        self
    }
    /// with_target_batch_size: with target batch size.
    pub fn with_target_batch_size(mut self, v: usize) -> Self {
        self.inner.target_batch_size = v;
        self
    }
    /// with_max_batch_size: with max batch size.
    pub fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    /// with_similarity_threshold: with similarity threshold.
    pub fn with_similarity_threshold(mut self, v: f32) -> Self {
        self.inner.similarity_threshold = v;
        self
    }
    /// build: build the [`SequencePackingConfig`].
    pub fn build(self) -> SequencePackingConfig {
        self.inner
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
    /// new: new.
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

impl SchedulerConfig {
    /// builder: construct via builder for documented field ergonomics.
    pub fn builder() -> SchedulerConfigBuilder {
        SchedulerConfigBuilder::default()
    }
}

/// Builder for [`SchedulerConfig`].
#[derive(Debug, Clone, Default)]
pub struct SchedulerConfigBuilder {
    inner: SchedulerConfig,
}

impl SchedulerConfigBuilder {
    /// with_max_num_seqs: with max num seqs.
    pub fn with_max_num_seqs(mut self, v: usize) -> Self {
        self.inner.max_num_seqs = v;
        self
    }
    /// with_max_num_batched_tokens: with max num batched tokens.
    pub fn with_max_num_batched_tokens(mut self, v: usize) -> Self {
        self.inner.max_num_batched_tokens = v;
        self
    }
    /// with_max_consecutive_decode: with max consecutive decode.
    pub fn with_max_consecutive_decode(mut self, v: u32) -> Self {
        self.inner.max_consecutive_decode = v;
        self
    }
    /// with_enable_pd_separation: with enable pd separation.
    pub fn with_enable_pd_separation(mut self, v: bool) -> Self {
        self.inner.enable_pd_separation = v;
        self
    }
    /// with_prefill_chunk_size: with prefill chunk size.
    pub fn with_prefill_chunk_size(mut self, v: usize) -> Self {
        self.inner.prefill_chunk_size = v;
        self
    }
    /// with_decode_preference_ratio: with decode preference ratio.
    pub fn with_decode_preference_ratio(mut self, v: f32) -> Self {
        self.inner.decode_preference_ratio = v;
        self
    }
    /// with_enable_priority_scheduling: with enable priority scheduling.
    pub fn with_enable_priority_scheduling(mut self, v: bool) -> Self {
        self.inner.enable_priority_scheduling = v;
        self
    }
    /// with_enable_dynamic_batching: with enable dynamic batching.
    pub fn with_enable_dynamic_batching(mut self, v: bool) -> Self {
        self.inner.enable_dynamic_batching = v;
        self
    }
    /// with_min_batch_size: with min batch size.
    pub fn with_min_batch_size(mut self, v: usize) -> Self {
        self.inner.min_batch_size = v;
        self
    }
    /// with_max_batch_size: with max batch size.
    pub fn with_max_batch_size(mut self, v: usize) -> Self {
        self.inner.max_batch_size = v;
        self
    }
    /// with_cuda_graph: with cuda graph.
    pub fn with_cuda_graph(mut self, v: SchedulerCudaGraphConfig) -> Self {
        self.inner.cuda_graph = v;
        self
    }
    /// with_packing: with packing.
    pub fn with_packing(mut self, v: SequencePackingConfig) -> Self {
        self.inner.packing = v;
        self
    }
    /// build: build the [`SchedulerConfig`].
    pub fn build(self) -> SchedulerConfig {
        self.inner
    }
}

use crate::metrics::MetricsSnapshot;

/// EngineMessage: engine message enumeration.
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
