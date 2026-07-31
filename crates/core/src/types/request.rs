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
/// external draft binding.
///
/// Constructed by the HTTP server (or any other caller) and submitted via
/// [`crate::engine::Engine::add_request`].
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

#[cfg(test)]
mod tests {
    // Exact-literal equality on f32 fields (0.0 / 1.0 defaults): the
    // literals are representable exactly in f32, so strict comparison
    // cannot fail on rounding.
    #![allow(clippy::float_cmp)]
    use super::*;

    #[test]
    fn new_sets_fields_from_arguments() {
        let req = Request::new(42, vec![1, 2, 3], 100);
        assert_eq!(req.id, 42);
        assert_eq!(req.prompt, vec![1, 2, 3]);
        assert_eq!(req.max_tokens, 100);
        // SamplingParams default has temperature = 0.0, top_p = 1.0 — verify
        // the key fields rather than full equality (SamplingParams doesn't
        // derive PartialEq since it comes from vllm_traits).
        assert_eq!(req.sampling_params.temperature, 0.0);
        assert_eq!(req.priority, Priority::default());
        assert_eq!(req.draft_model_id, None);
    }

    #[test]
    fn new_with_empty_prompt() {
        let req = Request::new(1, vec![], 50);
        assert!(req.prompt.is_empty());
        assert_eq!(req.max_tokens, 50);
    }

    #[test]
    fn with_priority_overrides_default() {
        let req = Request::new(0, vec![], 10).with_priority(Priority(5));
        assert_eq!(req.priority, Priority(5));
    }

    #[test]
    fn with_draft_model_sets_some() {
        let req = Request::new(0, vec![], 10).with_draft_model("draft-7b");
        assert_eq!(req.draft_model_id, Some(DraftId::from("draft-7b")));
    }

    #[test]
    fn builder_chain_compatible() {
        let req = Request::new(7, vec![10, 20], 200)
            .with_priority(Priority(3))
            .with_draft_model("spec-v2");
        assert_eq!(req.id, 7);
        assert_eq!(req.prompt, vec![10, 20]);
        assert_eq!(req.max_tokens, 200);
        assert_eq!(req.priority, Priority(3));
        assert_eq!(req.draft_model_id, Some(DraftId::from("spec-v2")));
    }

    #[test]
    fn priority_ord_is_total() {
        assert!(Priority(1) < Priority(2));
        assert!(Priority(0) == Priority(0));
        assert!(Priority(5) > Priority(4));
    }
}
