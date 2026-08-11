// crates/core/src/scheduler/batch_composer/compose/prefill.rs
//
// Prefill-phase composition. Two entry points:
// - `compose_prefill_with_packing`: packing-aware dispatcher (currently
//   delegates to standard prefill — see comment in body)
// - `compose_prefill_batch`: standard prefill batch builder
//
// Both produce a `Batch` with `BatchPhase::Prefill`. Sequences are sorted
// by remaining-token count (shorter first) so shorter sequences get
// included first when the token budget is tight (improves packing).

use super::BatchComposer;
use crate::types::{SamplingParams, Sequence};
use vllm_traits::{Batch, BatchPhase, BlockId, SeqId, TokenId};

/// Outcome of classifying a single sequence for prefill inclusion.
#[allow(clippy::large_enum_variant)]
enum PrefillAction {
    /// No new tokens to process (skip silently).
    Skip,
    /// Would exceed the token budget (stop the loop).
    Break,
    /// Include this sequence in the batch.
    Process {
        seq_id: SeqId,
        start: usize,
        positions: Vec<usize>,
        tokens: Vec<TokenId>,
        token_count: usize,
        kv_blocks: Vec<BlockId>,
        is_prefill: bool,
        sampling_params: SamplingParams,
    },
}

impl BatchComposer {
    /// Compose a prefill batch with sequence packing optimization.
    ///
    /// For now, this delegates to the standard prefill composer — the
    /// packing optimization that would reorder sequences to minimize
    /// padding is deferred. Behaviour is unchanged from
    /// `compose_prefill_batch`; the entry point exists so callers can
    /// opt into packing-aware composition without changing the public
    /// dispatcher.
    pub(super) fn compose_prefill_with_packing(&self, sequences: Vec<Sequence>) -> Batch {
        self.compose_prefill_batch(sequences)
    }

    /// Compose a prefill batch
    pub(super) fn compose_prefill_batch(&self, mut sequences: Vec<Sequence>) -> Batch {
        // Sort by remaining token count (shorter first for better packing)
        // H-13 (PERF-04): `sort_unstable_by_key` drops the stability
        // guarantee that `sort_by_key` provides. Stable ordering is not
        // relied on by downstream consumers of the prefill batch.
        sequences.sort_unstable_by_key(|s| s.tokens.len().saturating_sub(s.num_computed_tokens));

        // H-13 (PERF-03): pre-size the output vecs to `max_batch_size`
        // so the first `max_batch_size` pushes do not trigger
        // reallocation. Matches the decode-path pattern.
        let capacity = self.config.max_batch_size;
        let mut seq_ids = Vec::with_capacity(capacity);
        let mut input_tokens = Vec::with_capacity(capacity);
        let mut positions = Vec::with_capacity(capacity);
        let mut kv_block_ids = Vec::with_capacity(capacity);
        let mut num_computed_tokens = Vec::with_capacity(capacity);
        let mut is_prefill = Vec::with_capacity(capacity);
        let mut sampling_params = Vec::with_capacity(capacity);
        let mut total_tokens = 0usize;
        let mut max_seq_len = 0usize;

        tracing::debug!(
            sequences_count = sequences.len(),
            max_batch_size = self.config.max_batch_size,
            max_token_budget = self.config.max_token_budget,
            "compose_prefill: starting"
        );

        for seq in sequences.into_iter().take(self.config.max_batch_size) {
            tracing::debug!(seq_id = seq.id, "compose_prefill: processing sequence");

            match self.classify_prefill_seq(seq, total_tokens) {
                PrefillAction::Skip => {}
                PrefillAction::Break => break,
                PrefillAction::Process {
                    seq_id,
                    start,
                    positions: pos,
                    tokens,
                    token_count,
                    kv_blocks,
                    is_prefill: pref,
                    sampling_params: params,
                } => {
                    seq_ids.push(seq_id);
                    positions.push(pos);
                    total_tokens += token_count;
                    max_seq_len = max_seq_len.max(token_count);
                    input_tokens.push(tokens);
                    kv_block_ids.push(kv_blocks);
                    num_computed_tokens.push(start);
                    // Only treat as prefill if this is the first chunk
                    // of the sequence. If start > 0, this is a resume
                    // from partial prefill, use decode mode.
                    is_prefill.push(pref);
                    // ARCH-02: thread per-sequence sampling params into
                    // the batch so the engine applies them after
                    // `forward_logits`.
                    sampling_params.push(params);
                }
            }
        }

        tracing::debug!(
            batch_seq_count = seq_ids.len(),
            total_tokens = total_tokens,
            "compose_prefill: batch built"
        );

        Batch {
            seq_ids,
            input_tokens,
            positions,
            kv_block_ids,
            num_computed_tokens,
            is_prefill,
            sampling_params,
            phase: BatchPhase::Prefill,
            total_tokens,
            max_seq_len,
        }
    }

    /// Classify a sequence for prefill inclusion: skip (no new tokens),
    /// break (would exceed token budget), or process (include in batch).
    ///
    /// Extracts the per-sequence data (tokens, positions, `kv_blocks`, etc.)
    /// on the `Process` path so the caller's loop body stays flat.
    /// Tracing calls live here rather than in the hot loop.
    fn classify_prefill_seq(&self, seq: Sequence, total_tokens: usize) -> PrefillAction {
        let start = seq.num_computed_tokens;
        let seq_len = seq.tokens.len();
        let tokens_to_process = seq_len.saturating_sub(start);

        if tokens_to_process == 0 {
            tracing::debug!("Skipping: tokens_to_process == 0");
            return PrefillAction::Skip;
        }

        // RIL ISS-051: a sequence that overfills the batch token budget is
        // CHUNKED (process `budget_room` tokens now, resume the rest on a
        // later round) instead of breaking. Breaking meant a prompt longer
        // than `max_token_budget` could never be served — the queue spun in
        // Prefill forever, producing zero tokens, while the phase scheduler
        // kept selecting the (never-drained) prefill phase. Only
        // over-fitting prompts are chunked; prompts that fit run whole, so
        // mid-size prompt behaviour is unchanged. The chunk is capped by
        // `prefill_chunk_size` (the documented per-chunk token bound).
        let budget_room = self.config.max_token_budget.saturating_sub(total_tokens);
        let tokens_to_process = if tokens_to_process > budget_room {
            budget_room.min(self.chunked_prefill.target_chunk_size.max(1))
        } else {
            tokens_to_process
        };
        if tokens_to_process == 0 {
            // Batch budget already exhausted by earlier sequences.
            tracing::debug!(
                "Breaking: total_tokens {} reaches max_token_budget {}",
                total_tokens,
                self.config.max_token_budget
            );
            return PrefillAction::Break;
        }

        let chunk_end = start + tokens_to_process;
        let tokens: Vec<TokenId> = seq.tokens[start..chunk_end].to_vec();
        let token_count = tokens.len();
        if token_count < seq_len - start {
            tracing::debug!(
                seq_id = seq.id,
                start = start,
                chunk = token_count,
                remaining = seq_len - chunk_end,
                "chunked prefill: sequence exceeds the token budget; processing a chunk"
            );
        }
        PrefillAction::Process {
            seq_id: seq.id,
            start,
            positions: (start..chunk_end).collect(),
            tokens,
            token_count,
            kv_blocks: seq.kv_blocks.as_ref().clone(),
            // RIL ISS-021: a resumed prefill (start > 0 after a partial
            // prefix-cache hit) is still a prefill. It must keep
            // `is_prefill = true` so the model dispatches to `forward_prefill`,
            // which routes to `forward_prefill_continue` when
            // `num_computed_tokens > 0` (reading the cached KV and processing
            // the multi-token suffix with a rectangular causal mask). The old
            // `start == 0` sent the multi-token suffix to the single-token
            // `forward_decode` path, which reshapes to seq_len=1 and fails /
            // mis-processes any suffix longer than one token.
            is_prefill: true,
            sampling_params: seq.sampling_params,
        }
    }
}
