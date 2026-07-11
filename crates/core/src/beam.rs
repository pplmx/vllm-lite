//! Beam-search state types: one beam hypothesis with its accumulated
//! log-prob and emitted token sequence.
//!
//! `BeamSequence` is a public re-export ([`crate::BeamSequence`]). No
//! production scheduling code currently drives a beam search; the
//! state types are kept here so future work can wire beam search in
//! without re-deriving them.
#![allow(clippy::module_name_repetitions)]
use crate::types::{BlockId, TokenId};
use std::sync::Arc;

/// A single beam-search hypothesis: the generated token sequence, its
/// cumulative log-probability score, and the KV-cache blocks backing it.
///
/// `kv_blocks` is wrapped in an `Arc` so that forked beams (created when a
/// beam is expanded into multiple successors) can share the prefix KV cache
/// without copying. Only the divergent suffix needs new blocks.
#[derive(Clone, Debug)]
pub struct BeamSequence {
    pub tokens: Vec<TokenId>,
    pub score: f32,
    pub kv_blocks: Arc<Vec<BlockId>>,
}

impl BeamSequence {
    /// Construct a new beam from its initial tokens, score, and KV blocks.
    /// The block list is wrapped in an `Arc` so that subsequent forks can
    /// share it cheaply via `Arc::clone`.
    #[must_use]
    pub fn new(tokens: Vec<TokenId>, score: f32, kv_blocks: Vec<BlockId>) -> Self {
        Self {
            tokens,
            score,
            kv_blocks: Arc::new(kv_blocks),
        }
    }

    /// Extend the beam with a new token and add `log_prob` to the running
    /// score. Does not touch the KV-cache block list — the caller is
    /// responsible for appending the new block(s) when the model's forward
    /// pass over the appended token completes.
    pub fn push(&mut self, token: TokenId, log_prob: f32) {
        self.tokens.push(token);
        self.score += log_prob;
    }
}
