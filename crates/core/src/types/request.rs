//! Inbound generation requests.

use crate::speculative::DraftId;
use crate::types::sampling::SamplingParams;
use vllm_traits::{SeqId, TokenId};

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
