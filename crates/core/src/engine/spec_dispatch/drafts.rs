//! Draft generation strategies.
//!
//! Two implementations:
//! - [`generate_per_seq_drafts`](super::super::Engine::generate_per_seq_drafts) —
//!   v18.0 per-request dispatch via `DraftResolver`. Handles mixed routing
//!   and `degraded_draft` sticky fallback.
//! - [`generate_batched_drafts`](super::super::Engine::generate_batched_drafts) —
//!   Legacy batched single-draft path. All sequences share one draft model.

use crate::error::Result;
use crate::speculative::ResolvedDraft;
use crate::sync::lock_mutex;
use std::panic::{AssertUnwindSafe, catch_unwind};
use vllm_traits::{Batch, TokenId};

impl crate::engine::Engine {
    /// v18.0 per-request draft dispatch via `DraftResolver`.
    ///
    /// For each seq in the batch:
    /// 1. Skip if `sequence.degraded_draft` (FALL-02 sticky flag) — return
    ///    empty drafts so the seq falls through to non-spec decode.
    /// 2. Resolve via `resolver.resolve(seq.draft_model_id)` → picks the
    ///    named external draft, the self-spec fallback, or None.
    /// 3. Run the resolved backend's forward per position (up to `max_draft`).
    /// 4. Catch per-seq forward errors → set `degraded_draft = true` on that
    ///    sequence and increment `inc_draft_runtime_error`. Future steps skip
    ///    this seq's draft.
    ///
    /// Returns `Vec<Vec<TokenId>>` in the same shape as `generate_batched_drafts`
    /// so the downstream verification path is unchanged.
    pub(crate) fn generate_per_seq_drafts(
        &mut self,
        batch: &Batch,
        max_draft: usize,
    ) -> Vec<Vec<TokenId>> {
        let resolver = self
            .draft_resolver
            .as_ref()
            // invariant: caller contract — `generate_per_seq_drafts` is only invoked when a
            // `DraftResolver` has been installed (see `Engine::with_drafts_boxed` initialization).
            .expect("generate_per_seq_drafts called without draft_resolver");
        let n_seq = batch.seq_ids.len();
        let mut draft_outputs: Vec<Vec<TokenId>> = vec![Vec::with_capacity(max_draft); n_seq];

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            // Look up the sequence to check degraded_draft and read draft_model_id.
            let seq_state = self.scheduler.get_sequence(*seq_id);
            let (degraded, draft_model_id): (bool, Option<crate::speculative::DraftId>) =
                match seq_state {
                    Some(s) => (s.degraded_draft, s.draft_model_id.clone()),
                    None => {
                        // Sequence no longer in scheduler — skip (shouldn't happen
                        // in normal flow).
                        continue;
                    }
                };

            if degraded {
                // FALL-02 sticky: this sequence has been degraded. Skip draft
                // generation entirely; the seq falls through to non-spec
                // decode in the verifier path.
                continue;
            }

            let resolution = resolver.resolve(draft_model_id.as_ref());
            let backend = match resolution {
                ResolvedDraft::External(b) | ResolvedDraft::SelfSpec(b) => b,
                ResolvedDraft::None => continue,
            };

            // Per-position draft generation for this single sequence.
            let mut current_tokens: Vec<TokenId> = batch.input_tokens[i].clone();
            let mut current_positions: Vec<usize> = batch.positions[i].clone();
            for _pos in 0..max_draft {
                // `AssertUnwindSafe` is required because the closure captures
                // `backend: Arc<Mutex<Box<dyn ModelBackend>>>` along with the
                // `current_tokens`/`current_positions` buffers. The standard
                // `UnwindSafe` bound would refuse to compile even though the
                // catch is a deliberate guard against panics in foreign
                // backend code. If a panic occurs mid-forward, the backend's
                // internal state (KV cache, allocator) may be partially
                // mutated and the Mutex may become poisoned — subsequent
                // locks will themselves panic and be re-caught here. Note:
                // `catch_unwind` cannot catch aborts (e.g. double-panic,
                // `extern "C"` unwinding) — a misbehaving backend could still
                // crash the engine.
                let result = catch_unwind(AssertUnwindSafe(|| {
                    // invariant: lock is only held for synchronous field access; no panic possible while holding.
                    let mut guard = backend.lock().expect("draft backend mutex poisoned");
                    guard.forward(
                        &[*seq_id],
                        std::slice::from_ref(&current_tokens),
                        std::slice::from_ref(&current_positions),
                        std::slice::from_ref(&batch.kv_block_ids[i]),
                        std::slice::from_ref(&batch.num_computed_tokens[i]),
                        std::slice::from_ref(&batch.is_prefill[i]),
                    )
                }));

                #[allow(clippy::option_if_let_else)]
                let forward_result = match result {
                    Ok(r) => r,
                    Err(_) => {
                        // Panic in draft forward — treat as runtime error.
                        Err(vllm_traits::ModelError::new("draft forward panicked"))
                    }
                };

                match forward_result {
                    Ok(output) => {
                        if let Some(sampled) = output.next_tokens.first() {
                            let token = sampled.token;
                            draft_outputs[i].push(token);
                            current_tokens.push(token);
                            if let Some(&last_pos) = current_positions.last() {
                                current_positions.push(last_pos + 1);
                            }
                        } else {
                            // Empty output — stop draft generation for this seq.
                            break;
                        }
                    }
                    Err(e) => {
                        // FALL-02: draft forward error → mark degraded and skip
                        // future drafts for this seq.
                        tracing::warn!(
                            seq_id = %seq_id,
                            error = %e,
                            "draft forward failed; marking sequence degraded"
                        );
                        self.scheduler.metrics.inc_draft_runtime_error();
                        if let Some(s) = self.scheduler.get_sequence_mut(*seq_id) {
                            s.degraded_draft = true;
                        }
                        break;
                    }
                }
            }
        }

        draft_outputs
    }

    /// Batched per-position draft generation (Plan 17.1-B).
    ///
    /// All sequences generate draft position k before advancing to k+1.
    pub(crate) fn generate_batched_drafts(
        &self,
        batch: &Batch,
        max_draft: usize,
    ) -> Result<Vec<Vec<TokenId>>> {
        let n_seq = batch.seq_ids.len();
        let mut draft_outputs: Vec<Vec<TokenId>> = vec![Vec::with_capacity(max_draft); n_seq];

        let draft_model = if let Some(dm) = &self.draft_model {
            dm.clone()
        } else {
            tracing::warn!("No draft model set, returning empty drafts");
            return Ok(draft_outputs);
        };

        // Per-sequence state tracking: current input tokens and positions
        let mut current_tokens: Vec<Vec<TokenId>> = batch.input_tokens.iter().cloned().collect();
        let mut current_positions: Vec<Vec<usize>> = batch.positions.iter().cloned().collect();

        for pos in 0..max_draft {
            // Build per-position batch
            let mut pos_seq_ids = Vec::with_capacity(n_seq);
            let mut pos_input_tokens = Vec::with_capacity(n_seq);
            let mut pos_positions = Vec::with_capacity(n_seq);
            let mut pos_kv_block_ids = Vec::with_capacity(n_seq);
            let mut pos_num_computed = Vec::with_capacity(n_seq);
            let mut pos_is_prefill = Vec::with_capacity(n_seq);
            let mut active_indices = Vec::with_capacity(n_seq);

            for (i, seq_id) in batch.seq_ids.iter().enumerate() {
                if current_tokens[i].is_empty() {
                    continue;
                }
                pos_seq_ids.push(*seq_id);
                pos_input_tokens.push(current_tokens[i].clone());
                pos_positions.push(current_positions[i].clone());
                pos_kv_block_ids.push(batch.kv_block_ids[i].clone());
                pos_num_computed.push(batch.num_computed_tokens[i]);
                pos_is_prefill.push(if pos == 0 {
                    batch.is_prefill[i]
                } else {
                    false // subsequent draft steps are decode
                });
                active_indices.push(i);
            }

            if pos_seq_ids.is_empty() {
                break;
            }

            // Single forward pass per position across all active sequences
            let result = lock_mutex(&draft_model)?.forward(
                &pos_seq_ids,
                &pos_input_tokens,
                &pos_positions,
                &pos_kv_block_ids,
                &pos_num_computed,
                &pos_is_prefill,
            );
            let output = match result {
                Ok(o) => o,
                Err(e) => {
                    tracing::warn!(error = %e, pos = pos, "Draft model forward failed at position");
                    break;
                }
            };

            // Distribute output tokens back to per-sequence drafts
            for (j, &seq_idx) in active_indices.iter().enumerate() {
                let token = output.next_tokens.get(j).map_or(0, |sampled| sampled.token);
                draft_outputs[seq_idx].push(token);
                current_tokens[seq_idx].push(token);
                let new_pos = current_positions[seq_idx].len();
                current_positions[seq_idx].push(new_pos);
            }
        }

        Ok(draft_outputs)
    }
}

/// argmax: argmax helper used by the verifier path.
///
/// Tie-breaks to the FIRST max index to stay consistent with
/// [`vllm_traits::argmax_logits`] (the single source of truth for
/// greedy decoding). The two functions must agree on ties: the
/// verifier's temperature-aware path can fall through to either
/// depending on `params.temperature`, and a tie-break mismatch
/// would silently change the accepted draft sequence under
/// temperature == 0 when several logits share the maximum.
#[allow(dead_code)] // historical export; the verifier now delegates to sample_one_with_params directly (P36)
pub fn argmax(logits: &[f32]) -> TokenId {
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_idx = i;
            best_val = v;
        }
    }
    TokenId::try_from(best_idx).unwrap_or(0)
}
