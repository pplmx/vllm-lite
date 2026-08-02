//! Post-step state update for `SchedulerEngine`.
//!
//! `SchedulerEngine::update` is invoked after the model forward pass
//! returns. It folds newly generated tokens back into the running
//! sequences, advances their status (Prefill -> Decode -> Finished),
//! allocates additional KV blocks when the token count crosses a block
//! boundary, inserts finished sequences into the prefix cache, and
//! emits observer events.

use std::sync::Arc;

use vllm_traits::{BlockId, SampledToken, SeqId};

use crate::scheduler::observer::ObserverEvent;
use crate::types::Status;

use super::state::SchedulerEngine;

impl SchedulerEngine {
    /// Update the scheduler after model forward pass
    ///
    /// Processes output tokens, updates sequence status, handles completions,
    /// and adds finished sequences to the prefix cache.
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:**
    /// `next_tokens` carries [`SampledToken`] (token + logprob +
    /// `top_logprobs`) — only the `token` field is folded into the
    /// running sequence; the logprob + `top_logprobs` travel out
    /// through the engine's per-seq response channel alongside the
    /// token so the HTTP layer can render `OpenAI`'s
    /// `choices[].logprobs` shape.
    pub fn update(
        &mut self,
        seq_ids: &[SeqId],
        next_tokens: &[SampledToken],
        input_token_counts: &[usize],
    ) {
        let _span = tracing::info_span!(
            "scheduler.update",
            seq_count = seq_ids.len(),
            token_count = next_tokens.len()
        )
        .entered();

        tracing::debug!(
            seq_ids_len = seq_ids.len(),
            next_tokens_len = next_tokens.len(),
            input_counts_len = input_token_counts.len(),
            "Scheduler update"
        );
        for ((&seq_id, sampled), &input_count) in seq_ids
            .iter()
            .zip(next_tokens.iter())
            .zip(input_token_counts)
        {
            let token = sampled.token;
            let _token_span =
                tracing::trace_span!("scheduler.decode_token", seq_id = seq_id, token = token)
                    .entered();

            if let Some(idx) = self.running.iter().position(|s| s.id == seq_id) {
                self.update_running_sequence(idx, seq_id, token, input_count);
            }
        }

        self.finalize_finished_sequences();
    }

    /// Per-sequence update: status transition, token recording, observer
    /// dispatch, block allocation, and completion check.
    fn update_running_sequence(
        &mut self,
        idx: usize,
        seq_id: SeqId,
        token: u32,
        input_count: usize,
    ) {
        let Some(seq) = self.running.get_mut(idx) else {
            return;
        };
        tracing::debug!(
            seq_id = seq_id,
            tokens_len = seq.tokens.len(),
            status = ?seq.status,
            max_tokens = seq.max_tokens,
            "Scheduler update: processing sequence"
        );
        // Update status based on progress
        if seq.status == Status::Waiting || seq.status == Status::Prefilling {
            seq.num_computed_tokens += input_count;
            if seq.num_computed_tokens >= seq.prompt_len {
                seq.status = Status::Decoding;
                tracing::info!(seq_id = seq_id, "Sequence transitioned to Decode phase");
            } else {
                seq.status = Status::Prefilling;
            }
        }

        seq.tokens.push(token);
        seq.consecutive_decode_rounds += 1;

        // Dispatch observer event for token generation
        self.observers
            .dispatch(&ObserverEvent::Decoding { seq_id, token });

        // Allocate more blocks if needed
        let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
        while seq.kv_blocks.len() < blocks_needed {
            if let Some(new_blocks) = self.memory.allocate(1) {
                // ARCH-01 (technical due diligence): record the
                // freshly allocated blocks so the refcount
                // matches the number of live owners (this
                // sequence = 1). Without this, a subsequent
                // `release_blocks` from this sequence would
                // not actually return the block to the
                // allocator's free list — and worse, if the
                // prefix cache ever stored it, the block
                // would be freed underneath the cache.
                self.memory.record_blocks(&new_blocks);
                #[cfg(feature = "multi-node")]
                {
                    // Feed tokens for each newly-allocated block
                    // back to the MemoryManager so the chain
                    // hash advances with real content. Per-
                    // sequence cursor lives in `chain_cursors`;
                    // starting at `0` for the first block of
                    // each sequence (matches `BlockHasher`'s
                    // "parent_hash == 0 for first block" contract).
                    let block_idx = seq.kv_blocks.len();
                    let start = block_idx * vllm_traits::BLOCK_SIZE;
                    let end = (start + vllm_traits::BLOCK_SIZE).min(seq.tokens.len());
                    let parent_hash = self.chain_cursors.get(&seq_id).copied().unwrap_or(0);
                    for &block_id in &new_blocks {
                        let hash = self.memory.record_block_tokens(
                            block_id,
                            parent_hash,
                            &seq.tokens[start..end],
                        );
                        self.chain_cursors.insert(seq_id, hash);
                    }
                }
                let mut blocks = (*seq.kv_blocks).clone();
                blocks.extend(new_blocks);
                seq.kv_blocks = Arc::new(blocks);
            } else {
                break;
            }
        }

        // Check completion — max_tokens is the upper bound on *generated*
        // tokens (prompt not included), per the Request documentation and
        // the OpenAI API spec. Subtract prompt_len so the sequence finishes
        // after producing max_tokens output tokens rather than
        // max_tokens total tokens.
        if seq.tokens.len() - seq.prompt_len >= seq.max_tokens {
            seq.status = Status::Finished;
            // Add to prefix cache. ARCH-01: the prefix cache
            // now takes a reference to these blocks, so we
            // bump the refcount before the sequence releases
            // its own reference in the loop below. After this
            // insert the refcount is 2 (sequence + cache); the
            // subsequent release drops it to 1 (cache only),
            // keeping the block alive for the next prefix hit.
            let prompt_tokens = &seq.tokens[..seq.prompt_len];
            // Only the prompt-covering blocks are cached: decode blocks
            // hold this sequence's generated KV and are stale (they
            // would only ever be overwritten before read). Pinning them
            // previously inflated per-entry memory pressure — a few
            // long responses could pin the entire block pool.
            let prompt_blocks = seq
                .kv_blocks
                .len()
                .min(prompt_tokens.len().div_ceil(vllm_traits::BLOCK_SIZE));
            let blocks: Vec<BlockId> = seq.kv_blocks[..prompt_blocks].to_vec();
            self.memory.record_blocks(&blocks);
            let replaced = self.prefix_cache.insert(prompt_tokens, blocks);
            // Repeated prompts re-insert an existing cache entry; the
            // overwrite orphans the cache's previous refcount on the
            // replaced blocks. Release it so refcounts match live
            // owners — without this, every repeat completion inflated
            // the count, making the blocks unevictable and immune to
            // drain (RIL ISS-003). For a same-block overwrite this
            // nets to zero against the `record_blocks` above.
            if let Some(stale) = replaced {
                self.memory.release_blocks(stale.as_ref());
            }
        }
    }

    /// Move finished sequences out of `running` into `finished`,
    /// dispatching observer events and releasing KV blocks.
    fn finalize_finished_sequences(&mut self) {
        let finished_seqs: Vec<_> = self
            .running
            .iter()
            .filter(|s| s.status == Status::Finished)
            .cloned()
            .collect();

        for seq in &finished_seqs {
            self.observers.dispatch(&ObserverEvent::SequenceFinished {
                seq_id: seq.id,
                total_tokens: seq.tokens.len(),
            });
        }

        for seq in finished_seqs {
            self.memory.release_blocks(seq.kv_blocks.as_ref());
            self.finished.push(seq);
        }

        self.running.retain(|s| s.status != Status::Finished);
    }

    /// Mark a sequence as finished due to an external completion
    /// condition (e.g. a matched stop-sequence), releasing its KV
    /// blocks and moving it to the finished set.
    ///
    /// `update` only finalizes sequences that hit `max_tokens`; stop-
    /// matched sequences are detected separately in the engine's step
    /// loop and must be finalized here so they don't linger in `running`
    /// (status still `Decoding`). Without this, the sequence would be
    /// re-included in every subsequent batch — wasting compute on
    /// tokens the client will never see and leaking KV cache blocks
    /// until `max_tokens` is eventually reached.
    pub fn finish_sequence(&mut self, seq_id: SeqId) {
        // invariant: position search is bounded by running.len(), which
        // is capped at config.max_running (256 default).
        let Some(idx) = self.running.iter().position(|s| s.id == seq_id) else {
            return;
        };
        let mut seq = self.running.remove(idx);
        seq.status = Status::Finished;
        self.observers.dispatch(&ObserverEvent::SequenceFinished {
            seq_id: seq.id,
            total_tokens: seq.tokens.len(),
        });
        self.memory.release_blocks(seq.kv_blocks.as_ref());
        self.finished.push(seq);
    }
}
