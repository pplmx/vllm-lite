//! Post-step state update for `SchedulerEngine`.
//!
//! `SchedulerEngine::update` is invoked after the model forward pass
//! returns. It folds newly generated tokens back into the running
//! sequences, advances their status (Prefill -> Decode -> Finished),
//! allocates additional KV blocks when the token count crosses a block
//! boundary, inserts finished sequences into the prefix cache, and
//! emits observer events.

use std::sync::Arc;

use vllm_traits::{BlockId, SeqId, TokenId};

use crate::scheduler::observer::ObserverEvent;
use crate::types::Status;

use super::state::SchedulerEngine;

impl SchedulerEngine {
    /// Update the scheduler after model forward pass
    ///
    /// Processes output tokens, updates sequence status, handles completions,
    /// and adds finished sequences to the prefix cache.
    pub fn update(
        &mut self,
        seq_ids: &[SeqId],
        next_tokens: &[TokenId],
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
        for ((&seq_id, &token), &input_count) in
            seq_ids.iter().zip(next_tokens).zip(input_token_counts)
        {
            let _token_span =
                tracing::trace_span!("scheduler.decode_token", seq_id = seq_id, token = token)
                    .entered();

            if let Some(seq) = self.running.iter_mut().find(|s| s.id == seq_id) {
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

                // Check completion
                if seq.tokens.len() >= seq.max_tokens {
                    seq.status = Status::Finished;
                    // Add to prefix cache
                    let prompt_tokens = &seq.tokens[..seq.prompt_len];
                    let blocks: Vec<BlockId> = seq.kv_blocks.as_ref().clone();
                    self.prefix_cache.insert(prompt_tokens, blocks);
                }
            }
        }

        // Collect finished sequences and dispatch observer events
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
}
