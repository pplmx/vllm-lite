//! Inbound generation requests.

use crate::speculative::DraftId;
use crate::types::sampling::SamplingParams;
use vllm_traits::{SeqId, TokenId};

/// Request priority — higher numeric value = higher priority. Used by the
/// scheduler when `enable_priority_scheduling` is `true`. Wraps a `u8` so it
/// stays small and `Ord`-friendly.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Priority(pub u8);

/// An inbound generation request: prompt + sampling configuration + optional
/// external draft binding. Constructed by the HTTP server (or any other
/// caller) and submitted via [`crate::engine::Engine::add_request`].
#[derive(Clone, Debug)]
pub struct Request {
    /// Caller-supplied identifier for correlating logs and responses. Not
    /// used as the internal sequence id; the engine assigns its own.
    pub id: SeqId,
    /// Already-tokenized prompt. Empty prompts are rejected by the engine.
    pub prompt: Vec<TokenId>,
    /// Upper bound on generated tokens (prompt not included). The engine
    /// stops the sequence once it has produced this many tokens.
    pub max_tokens: usize,
    /// Sampling configuration (temperature, top-k, top-p, repeat penalty,
    /// beam width). See [`SamplingParams`].
    pub sampling_params: SamplingParams,
    /// Scheduling priority. Honored only when priority scheduling is enabled
    /// in [`crate::SchedulerConfig`].
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
    /// Create a request with default sampling parameters and no draft
    /// binding. Use the `with_*` builder methods to customize.
    #[must_use]
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

    /// Override the request's scheduling priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Bind this request to a specific external draft model. The engine will
    /// resolve `id` against the registry at step time.
    #[must_use]
    pub fn with_draft_model(mut self, id: impl Into<DraftId>) -> Self {
        self.draft_model_id = Some(id.into());
        self
    }
}
