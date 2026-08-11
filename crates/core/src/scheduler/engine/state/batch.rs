//! `SchedulerEngine::build_batch` + `schedule` — phase selection, batch
//! composition, preemption trigger, and CUDA Graph / observer hooks.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use vllm_traits::{Batch, BlockId, SeqId};

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

        // RIL ISS-052: for a Decode round, grow each running `Decoding`
        // sequence's KV table to cover its current tokens — the preemption
        // gate above freed blocks, but freeing alone doesn't extend the
        // sequence's OWN table (growth in `update()` runs only AFTER a
        // successful forward, so a freed block is never assigned to the
        // short table in time). Defer any sequence that still cannot obtain
        // its growth block so it is never forwarded with a short table.
        if phase == Phase::Decode {
            sequences = self.admit_decode_sequences();
        }

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

    /// Recovery after a step error (RIL ISS-045): roll back any sequence
    /// left stranded in `running` before its prefill completed.
    ///
    /// A model-forward error in the engine's step propagates with `?`
    /// before `update()` ever runs, so the freshly-admitted sequences stay
    /// in `running` as `Prefilling` (fresh) or `Waiting` (preempted-resume)
    /// with their KV blocks pinned. `build_batch` only re-includes
    /// `Decoding` sequences from `running` and pulls prefill work from the
    /// *queue*, so they are never re-scheduled — the scheduler spins forever
    /// with `has_pending()` true, the KV blocks leak, and the client's
    /// finish channel never fires. Release and re-queue them so the request
    /// retries on a later round.
    ///
    /// The invariant "`running` only holds `Decoding` across steps" holds in
    /// healthy operation: `update()` transitions every admitted sequence to
    /// `Decoding` (or `Finished`) before the step returns, so any leftover
    /// `Prefilling` / `Waiting` after a step is exactly the orphaned set.
    pub(crate) fn requeue_stuck_prefills(&mut self) {
        let mut i = 0;
        while i < self.running.len() {
            let status = self.running[i].status;
            if status == Status::Prefilling || status == Status::Waiting {
                let seq = self.running.remove(i);
                self.requeue_seq(seq);
            } else {
                i += 1;
            }
        }
    }

    /// RIL ISS-052: select the `Decoding` sequences for this round's batch,
    /// ensuring each one's KV table covers its current `tokens` first.
    ///
    /// A decode sequence's table grows in `update()` *after* a successful
    /// forward, so when a boundary-crossing allocation fails (pool
    /// exhausted) it is left one block short exactly when the next forward
    /// writes the new token's KV — the paged-KV fallback would write into
    /// block 0 and corrupt the prompt's first block. The preemption gate in
    /// `build_batch` freed blocks but never assigns them to the sequence's
    /// own table. This grows the originals (in place, so `update()` keeps
    /// progressing) using the freed pool, then clones them; a sequence that
    /// still cannot obtain its growth block is deferred — it stays
    /// `Decoding` in `running` and is re-included on a later round — rather
    /// than being forwarded with a short table.
    fn admit_decode_sequences(&mut self) -> Vec<Sequence> {
        let mut admitted = Vec::with_capacity(self.running.len());
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].status != Status::Decoding {
                i += 1;
                continue;
            }
            let needed = self.running[i]
                .tokens
                .len()
                .div_ceil(vllm_traits::BLOCK_SIZE);
            if self.running[i].kv_blocks.len() < needed {
                let have = self.running[i].kv_blocks.len();
                self.grow_running_table(i, have, needed);
            }
            if self.running[i].kv_blocks.len() >= needed {
                admitted.push(self.running[i].clone());
            }
            i += 1;
        }
        admitted
    }

    /// Grow `self.running[idx]`'s KV table from `have` to `needed` blocks,
    /// allocating from the pool the preemption gate just tried to free up.
    /// Partial success leaves the table short — the caller decides whether
    /// to defer the sequence. Mirrors `preallocate_kv_blocks`'s
    /// allocate + `record_blocks` + extend pattern (and its multi-node
    /// token-hash recording) for the running-decode case.
    fn grow_running_table(&mut self, idx: usize, have: usize, needed: usize) {
        let mut additions: Vec<BlockId> = Vec::with_capacity(needed - have);
        while additions.len() < needed - have {
            match self.memory.allocate(1) {
                Some(new_blocks) => additions.extend(new_blocks),
                None => break,
            }
        }
        if additions.is_empty() {
            return;
        }
        self.memory.record_blocks(&additions);
        #[cfg(feature = "multi-node")]
        {
            let seq_id = self.running[idx].id;
            for (offset, &block_id) in additions.iter().enumerate() {
                let block_idx = have + offset;
                let start = block_idx * vllm_traits::BLOCK_SIZE;
                let end = (start + vllm_traits::BLOCK_SIZE).min(self.running[idx].tokens.len());
                let parent_hash = self.chain_cursors.get(&seq_id).copied().unwrap_or(0);
                let hash = self.memory.record_block_tokens(
                    block_id,
                    parent_hash,
                    &self.running[idx].tokens[start..end],
                );
                self.chain_cursors.insert(seq_id, hash);
            }
        }
        let mut blocks = (*self.running[idx].kv_blocks).clone();
        blocks.extend(additions);
        self.running[idx].kv_blocks = Arc::new(blocks);
    }

    /// Release a freshly-admitted (never-forwarded) sequence's pre-allocated KV
    /// blocks and return it to the waiting queue as `Waiting::Prefill` so it is
    /// retried on a later round instead of stalling in `running` or serving a
    /// corrupt KV table. Used for prefill overflow (ISS-041), partial-block
    /// OOM admission (ISS-042), and forward-error recovery (ISS-045) — all
    /// of which only ever requeue sequences that have not been advanced.
    ///
    /// RIL ISS-054: an ADVANCED sequence (a chunked prefill that made partial
    /// progress before being requeued) must keep its frontier +
    /// already-computed KV blocks — its block table is the full prompt table
    /// from admission, so it is complete and safe to resume. The old
    /// unconditional reset recomputed a long prompt from scratch on every
    /// contention round (ISS-041 overflow) or every transient forward error
    /// (ISS-045) — work-loss thrashing under load. Only never-advanced
    /// sequences (fresh admission whose preallocation failed) are reset.
    fn requeue_seq(&mut self, mut seq: Sequence) {
        if seq.num_computed_tokens > 0 {
            tracing::debug!(
                seq_id = seq.id,
                num_computed = seq.num_computed_tokens,
                "requeue: preserving advanced prefill's computed KV (chunked prefill)"
            );
            seq.status = Status::Waiting;
            let ctx = SchedulingContext {
                current_time: Instant::now(),
                queue_length: self.request_queue.len(),
                running_count: self.running.len(),
                memory_pressure: self.get_memory_pressure(),
            };
            self.request_queue.enqueue(seq, self.policy.as_ref(), &ctx);
            return;
        }
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
