//! Draft KV cache warmup after prefill.
//!
//! Invoked once per `step_speculative_inner` call when the batch is in
//! `Prefill` phase. Ensures the draft model's KV cache is populated before
//! the first speculative decode step, avoiding cold-start penalties.

use crate::error::Result;
use crate::speculative::ResolvedDraft;
use crate::sync::lock_mutex;
use vllm_traits::Batch;

impl crate::engine::Engine {
    /// Warm up draft model's KV cache after target prefill.
    /// This ensures the first speculative decode step has valid draft state.
    ///
    /// v18.0: when a draft resolver is installed, each seq's draft is
    /// resolved individually via the resolver so the warmup matches the
    /// backend that `generate_per_seq_drafts` will actually invoke. For
    /// `ResolvedDraft::None`, the warmup is skipped (pure target decode).
    /// When no resolver is installed, falls back to the legacy single-
    /// `draft_model` behavior for backward compatibility with `new_boxed`.
    pub(crate) fn warmup_draft_kv(&mut self, batch: &Batch) -> Result<()> {
        if !self.speculative_mode {
            return Ok(());
        }
        if let Some(resolver) = self.draft_resolver.clone() {
            // v18.0 per-seq warmup: resolve each seq's draft and warm it.
            for (i, seq_id) in batch.seq_ids.iter().enumerate() {
                let draft_model_id = self
                    .scheduler
                    .get_sequence(*seq_id)
                    .and_then(|s| s.draft_model_id.clone());
                let resolved = resolver.resolve(draft_model_id.as_ref());
                let backend = match resolved {
                    ResolvedDraft::External(b) | ResolvedDraft::SelfSpec(b) => b,
                    ResolvedDraft::None => continue,
                };
                lock_mutex(&backend)?.forward(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
            }
            tracing::debug!(
                seq_count = batch.seq_ids.len(),
                "Per-seq draft KV cache warmed up after prefill (v18.0)"
            );
        } else {
            // Legacy path: warm the single self.draft_model, if any.
            let draft_model: std::sync::Arc<std::sync::Mutex<Box<dyn vllm_traits::ModelBackend>>> =
                match &self.draft_model {
                    Some(dm) => dm.clone(),
                    None => return Ok(()),
                };
            for (i, seq_id) in batch.seq_ids.iter().enumerate() {
                lock_mutex(&draft_model)?.forward(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
            }
            tracing::debug!(
                seq_count = batch.seq_ids.len(),
                "Draft KV cache warmed up after prefill (legacy)"
            );
        }
        Ok(())
    }
}
