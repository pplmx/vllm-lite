//! Self-speculation implementation using reduced layer count
//!
//! This module provides self-speculation where the draft model uses
//! a subset of layers from the target model, sharing weights (zero copy).

// crates/core/src/speculative/self_spec/mod.rs
//
// Struct + constructor + accessors. The DraftVerifier trait impl lives
// in `verifier.rs`; tests live in `tests.rs`.

use crate::speculative::config::SpeculationConfig;
use crate::types::SeqId;
use std::collections::HashMap;
use vllm_traits::ModelBackend;

mod verifier;

#[derive(Debug)]
/// `SelfSpeculativeModel`. See the type definition for fields and behavior.
pub struct SelfSpeculativeModel<M: ModelBackend> {
    model: M,
    draft_layer_count: usize,
    draft_kv_block_ids: HashMap<SeqId, Vec<usize>>,
}

impl<M: ModelBackend> SelfSpeculativeModel<M> {
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(model: M, config: SpeculationConfig) -> Self {
        let total_layers = model.num_layers();
        // invariant: total_layers is a small model-architecture constant; the
        // `.max(1.0)` ensures the result is non-negative, so the f32 -> usize
        // cast is sign-safe.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let draft_layer_count = config
            .draft_layers
            .unwrap_or_else(|| (total_layers as f32 * 0.125).max(1.0) as usize);
        Self {
            model,
            draft_layer_count,
            draft_kv_block_ids: HashMap::new(),
        }
    }

    pub const fn model(&self) -> &M {
        &self.model
    }

    pub const fn mut_model(&mut self) -> &mut M {
        &mut self.model
    }

    pub const fn draft_layer_count(&self) -> usize {
        self.draft_layer_count
    }

    pub const fn set_draft_layer_count(&mut self, count: usize) {
        self.draft_layer_count = count;
    }

    pub const fn draft_kv_block_ids(&self) -> &HashMap<SeqId, Vec<usize>> {
        &self.draft_kv_block_ids
    }

    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn remove_draft_seq(&mut self, seq_id: SeqId) {
        self.draft_kv_block_ids.remove(&seq_id);
    }
}

#[cfg(test)]
mod tests;
