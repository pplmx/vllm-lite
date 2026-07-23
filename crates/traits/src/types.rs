//! Wire-format identifier types and constants shared by every backend and the scheduler.
//!
//! `SeqId`, `TokenId`, `BlockId` are opaque newtype aliases so that integer
//! literals at call sites stay readable (`seq_id: SeqId = 42` instead of
//! `seq_id: u64 = 42`); `BLOCK_SIZE` is the paged-KV allocator's page size.
use serde::{Deserialize, Serialize};

pub use crate::sampling::SampledToken;

use crate::sampling::SamplingParams;

/// Compile-time constant: `size`. Tune via feature flags or env vars in production.
pub const BLOCK_SIZE: usize = 16;
/// Opaque newtype identifier for a block. Hashable, comparable, serializable; use this rather than the raw integer.
pub type BlockId = usize;
/// Opaque newtype identifier for a token. Hashable, comparable, serializable; use this rather than the raw integer.
pub type TokenId = u32;
/// Opaque newtype identifier for a seq. Hashable, comparable, serializable; use this rather than the raw integer.
pub type SeqId = u64;

/// Batch phase classification for mixed prefill/decode scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchPhase {
    /// All sequences are in prefill.
    Prefill,
    /// All sequences are in decode.
    Decode,
    /// Batch contains both prefill and decode sequences.
    Mixed,
}

/// Why a generation request stopped.
///
/// Mirrors the `OpenAI` `finish_reason` semantics for the cases vLLM-lite
/// can actually produce today. The HTTP layer in `vllm-server` maps
/// these to the corresponding `OpenAI` string values
/// (`"stop"` / `"length"`).
///
/// `Cancelled` is emitted when the request was aborted by
/// `EngineMessage::CancelRequest` (e.g. the SSE client disconnected
/// mid-stream and the `CancelOnDrop` guard fired) — the HTTP layer
/// surfaces this by truncating the response rather than emitting a
/// final chunk, since the stream is already gone.
///
/// See `docs/technical-due-diligence/architecture-performance.md` §5.1.2
/// and the v31.0 P4 follow-up batch — pre-fix the engine closed the
/// response channel without telling the handler whether the stop was
/// natural, hit-max-tokens, or cancelled, and the HTTP layer
/// hardcoded `finish_reason = "stop"` for all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Model emitted an end-of-sentence token, or a configured stop
    /// sequence matched.
    Stop,
    /// Sequence hit its `max_tokens` cap without producing an EOS /
    /// stop-sequence token.
    Length,
    /// Sequence was cancelled by an external request (client
    /// disconnect, admin shutdown, etc.). The HTTP layer does not
    /// surface this in the response body — the stream is already
    /// gone by the time the engine reports it.
    Cancelled,
}

/// One batched inference step: a list of sequences together with the
/// per-sequence input tokens and positional metadata the scheduler
/// needs to issue prefill and decode work concurrently.
///
/// Serialized to JSON when a batch is checkpointed to the request log.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    /// Identifiers of every sequence participating in this batch.
    pub seq_ids: Vec<SeqId>,
    /// Per-sequence input tokens (prefill) or last-known token (decode).
    pub input_tokens: Vec<Vec<TokenId>>,
    /// Per-sequence absolute positions (used for `RoPE`).
    pub positions: Vec<Vec<usize>>,
    // KV cache information
    /// Per-sequence KV-cache block ids (parallel to `seq_ids`).
    pub kv_block_ids: Vec<Vec<BlockId>>,
    /// How many tokens have already been computed and stored in the KV cache.
    pub num_computed_tokens: Vec<usize>,
    /// Per-sequence `true` for prefill, `false` for decode.
    pub is_prefill: Vec<bool>,
    /// Per-sequence sampling parameters (temperature, top-p, top-k,
    /// repeat penalty, etc.). The engine reads this together with
    /// `forward_logits` output to produce the next token; an empty
    /// `sampling_params` (e.g. on a synthetic test batch) is
    /// equivalent to greedy decoding.
    ///
    /// ARCH-02 (technical due diligence): carrying the params on
    /// the batch closes the loop where the HTTP layer accepted them
    /// but the model layer always did greedy sampling. The model
    /// layer used to own the sampling decision; that decision now
    /// moves up to the engine where the params are visible.
    pub sampling_params: Vec<SamplingParams>,
    // New fields
    /// Overall phase classification for this batch.
    pub phase: BatchPhase,
    /// Sum of `input_tokens.len()` across all sequences.
    pub total_tokens: usize,
    /// Longest sequence in this batch; used for padding / kernel launches.
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
            sampling_params: Vec::new(),
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

    /// Whether any sequence in this batch is in prefill phase.
    #[must_use]
    pub fn has_prefill(&self) -> bool {
        self.is_prefill.iter().any(|&p| p)
    }

    /// Whether any sequence in this batch is in decode phase.
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

/// Output from Batch: the result payload plus any associated metadata. Returned to the caller.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchOutput {
    /// Identifiers of the sequences that produced these tokens (parallel to `next_tokens`).
    pub seq_ids: Vec<SeqId>,
    /// Per-sequence sampled tokens emitted by the model + sampler
    /// (P36 v0.3 wire-type follow-up engine wire-through). Each
    /// [`SampledToken`] carries the sampled `token` alongside its
    /// `logprob` under the post-filter distribution and (when
    /// `Batch.sampling_params[i].top_logprobs.is_some()`) the
    /// top-K `(token, logprob)` pairs.
    pub next_tokens: Vec<SampledToken>,
}

/// Error type for `TensorParallel`. Returned from every fallible public API; covers I/O, validation, and resource-limit failures. Use [`Result<T>`] alias in the same module.
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
