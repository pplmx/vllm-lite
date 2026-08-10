//! `SchedulerEngine::build_batch` + `schedule` — phase selection, batch
//! composition, preemption trigger, and CUDA Graph / observer hooks.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use vllm_traits::{Batch, SeqId};

use super::SchedulerEngine;
use crate::scheduler::SchedulerState;
use crate::scheduler::observer::ObserverEvent;
use crate::scheduler::policy::SchedulingContext;
use crate::types::{Phase, Sequence, Status};

impl SchedulerEngine {
    /// Build the next batch of sequences to process
    ///
    /// Uses the phase scheduler to determine whether to build a prefill or decode batch,
    /// then composes the batch according to memory constraints.
    #[must_use]
    pub fn build_batch(&mut self) -> Batch {
        let _span = tracing::info_span!(
            "scheduler.build_batch",
            waiting = self.request_queue.len(),
            running = self.running.len()
        )
        .entered();

        let start_time = Instant::now();

        // Get current scheduler state
        let state = SchedulerState {
            waiting_count: self.request_queue.len(),
            running_count: self.running.len(),
            prefill_queue_len: self.request_queue.phase_len(Phase::Prefill),
            decode_queue_len: self.request_queue.phase_len(Phase::Decode),
            available_memory: self.memory.available_blocks(),
            consecutive_decode_rounds: 0,
        };

        let phase = self.phase_scheduler.select_phase(&state);

        // Only include running decode sequences when in Decode phase
        let mut sequences: Vec<crate::types::Sequence> = if phase == Phase::Decode {
            self.running
                .iter()
                .filter(|s| s.status == Status::Decoding)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        // Get sequences for this phase from the queue
        let mut new_sequences = self.request_queue.drain_by_phase(phase);

        // Update metrics: queue depth after draining
        self.metrics
            .set_queue_depth(self.request_queue.len() as u64);

        // If no running decode sequences and no new sequences, return empty
        if sequences.is_empty() && new_sequences.is_empty() {
            return Batch::empty();
        }

        // Check memory and preempt if needed (before allocating the new
        // sequences' KV blocks below). Considers both the running decode
        // sequences and the newly-admitted ones.
        for seq in sequences.iter().chain(new_sequences.iter()) {
            let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
            if blocks_needed > self.memory.available_blocks() {
                self.execute_preemption(blocks_needed);
            }
        }

        // RIL ISS-022: allocate the KV blocks for newly-admitted sequences
        // BEFORE the forward pass writes their KV. The model's prefill writes
        // KV to `seq.kv_blocks`; if blocks are not allocated yet, the write
        // falls back to block 0 and corrupts the cache. (Previously blocks
        // were allocated only in `update()`, AFTER the forward, so the first
        // prefill wrote its KV to the wrong blocks.) Decode sequences still
        // grow incrementally in `update()` as they cross block boundaries.
        self.preallocate_kv_blocks(&mut new_sequences);

        // ISS-042: do not admit a sequence whose KV block table could not be
        // fully pre-allocated (see the helper for the full rationale).
        new_sequences = self.filter_fully_allocated_prefills(new_sequences, phase);

        // Add new sequences to the batch (now with allocated KV blocks)
        sequences.extend(new_sequences.iter().cloned());

        // Sort by policy priority
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: self.request_queue.len(),
            running_count: self.running.len(),
            memory_pressure: self.get_memory_pressure(),
        };

        sequences.sort_by(|a, b| {
            let priority_a = self.policy.compute_priority(a, &ctx);
            let priority_b = self.policy.compute_priority(b, &ctx);
            priority_a.cmp(&priority_b)
        });

        // Track which freshly-admitted prefill sequences were moved to running,
        // so we can re-queue any of them that `compose` below does not serve.
        let admitted: HashSet<SeqId> = new_sequences.iter().map(|s| s.id).collect();
        // Move new sequences to running
        self.running.extend(new_sequences);

        // Update metrics: active sequences
        self.metrics.set_active_sequences(self.running.len() as u64);

        // Build the batch
        let batch = self.batch_composer.compose(sequences.clone(), phase);

        // ISS-041: `compose` only serves `max_batch_size` / the token budget;
        // requeue the admitted-but-uncomposed prefill overflow (see the
        // helper for the full rationale).
        self.requeue_uncomposed_prefills(&admitted, &batch, phase);

        // Record CUDA Graph metrics if applicable
        if phase == Phase::Decode && self.cuda_graph.enabled {
            let batch_size = batch.seq_ids.len();
            if self.cuda_graph.supports_batch_size(batch_size) {
                self.metrics.record_cuda_graph_hit();
            } else {
                self.metrics.record_cuda_graph_miss();
            }
        }

        // Dispatch observer event
        if !batch.seq_ids.is_empty() {
            self.observers.dispatch(&ObserverEvent::BatchScheduled {
                seq_ids: batch.seq_ids.clone(),
                batch_size: batch.seq_ids.len(),
            });
        }

        // Record batch scheduling latency
        let duration = start_time.elapsed();
        self.metrics
            .record_inference_latency(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));

        let prefill_count = batch.is_prefill.iter().filter(|&&x| x).count();
        tracing::debug!(
            batch_size = batch.seq_ids.len(),
            prefill_count = prefill_count,
            total_tokens = batch.total_tokens,
            phase = ?batch.phase,
            "Batch built"
        );

        batch
    }
}

impl SchedulerEngine {
    /// Allocate KV blocks for freshly-admitted sequences BEFORE the forward
    /// pass writes their KV (RIL ISS-022).
    ///
    /// Shared by both batch builders (`build_batch` and
    /// `build_batch_with_graph`): the model's prefill writes KV to
    /// `seq.kv_blocks`, and if blocks are not allocated yet, `write_prefill_kv`
    /// falls back to block 0 and corrupts the cache. The CUDA-Graph batch
    /// builder previously skipped this step, corrupting every prefill on
    /// `cuda-graph` builds (RIL ISS-028).
    pub(crate) fn preallocate_kv_blocks(&mut self, seqs: &mut [Sequence]) {
        for seq in seqs.iter_mut() {
            let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
            while seq.kv_blocks.len() < blocks_needed {
                if let Some(new_blocks) = self.memory.allocate(1) {
                    // ARCH-01: record the freshly allocated blocks so the
                    // refcount matches the live owners (this sequence = 1).
                    self.memory.record_blocks(&new_blocks);
                    #[cfg(feature = "multi-node")]
                    {
                        let block_idx = seq.kv_blocks.len();
                        let start = block_idx * vllm_traits::BLOCK_SIZE;
                        let end = (start + vllm_traits::BLOCK_SIZE).min(seq.tokens.len());
                        let parent_hash = self.chain_cursors.get(&seq.id).copied().unwrap_or(0);
                        for &block_id in &new_blocks {
                            let hash = self.memory.record_block_tokens(
                                block_id,
                                parent_hash,
                                &seq.tokens[start..end],
                            );
                            self.chain_cursors.insert(seq.id, hash);
                        }
                    }
                    let mut blocks = (*seq.kv_blocks).clone();
                    blocks.extend(new_blocks);
                    seq.kv_blocks = Arc::new(blocks);
                } else {
                    break;
                }
            }
        }
    }

    /// ISS-042: drop any prefill sequence whose KV block table could not be
    /// fully pre-allocated. With the pool exhausted even after the preemption
    /// gate, serving such a sequence would let `write_prefill_kv` fall back to
    /// block 0 and corrupt the cache (see ISS-026/028). Defer: release the
    /// partial blocks, reset to Waiting-Prefill, and re-queue so it retries
    /// once memory frees. Only prefill needs the whole table up front; decode
    /// sequences grow blocks incrementally in `update()`.
    fn filter_fully_allocated_prefills(
        &mut self,
        new_sequences: Vec<Sequence>,
        phase: Phase,
    ) -> Vec<Sequence> {
        if phase != Phase::Prefill {
            return new_sequences;
        }
        let mut admissible = Vec::with_capacity(new_sequences.len());
        for seq in new_sequences {
            let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
            if seq.kv_blocks.len() < blocks_needed {
                self.requeue_seq(seq);
            } else {
                admissible.push(seq);
            }
        }
        admissible
    }

    /// ISS-041: `compose` includes at most `max_batch_size` sequences and
    /// breaks once the token budget is exceeded, but every drained prefill was
    /// already moved into `running` above. A freshly-admitted prefill that was
    /// NOT composed would be left `Status::Prefilling` in `running`, where it
    /// is never re-scheduled (running is only re-included when
    /// `status == Decoding`; once the waiting queue empties the Prefill phase
    /// is never selected again) — a permanent stall that also pins its
    /// pre-allocated KV blocks. Release and re-queue the overflow so it is
    /// retried on a later round. Decode overflow is harmless because
    /// `Decoding` sequences are re-included on the next decode round.
    fn requeue_uncomposed_prefills(
        &mut self,
        admitted: &HashSet<SeqId>,
        batch: &vllm_traits::Batch,
        phase: Phase,
    ) {
        if phase != Phase::Prefill {
            return;
        }
        let composed: HashSet<SeqId> = batch.seq_ids.iter().copied().collect();
        for seq_id in admitted {
            if composed.contains(seq_id) {
                continue;
            }
            if let Some(pos) = self.running.iter().position(|s| s.id == *seq_id) {
                let seq = self.running.remove(pos);
                self.requeue_seq(seq);
            }
        }
    }

    /// Release a freshly-admitted (never-forwarded) sequence's pre-allocated KV
    /// blocks and return it to the waiting queue as `Waiting::Prefill` so it is
    /// retried on a later round instead of stalling in `running` or serving a
    /// corrupt KV table. Used for prefill overflow (ISS-041) and partial-block
    /// OOM admission (ISS-042), both of which only ever requeue sequences that
    /// were admitted this round and never forwarded.
    fn requeue_seq(&mut self, mut seq: Sequence) {
        self.memory.release_blocks(seq.kv_blocks.as_ref());
        seq.kv_blocks = Arc::new(vec![]);
        seq.status = Status::Waiting;
        seq.num_computed_tokens = 0;
        let ctx = SchedulingContext {
            current_time: Instant::now(),
            queue_length: self.request_queue.len(),
            running_count: self.running.len(),
            memory_pressure: self.get_memory_pressure(),
        };
        self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);
    }
}
