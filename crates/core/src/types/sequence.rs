//! Active decode sequence state.

use std::sync::Arc;

use crate::speculative::DraftId;
use crate::types::request::Priority;
use crate::types::sampling::SamplingParams;
use vllm_traits::{BlockId, SeqId, TokenId};

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
