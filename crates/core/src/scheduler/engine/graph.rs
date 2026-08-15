//! Graph helper methods for `SchedulerEngine`.
//!
//! These methods drive the CUDA Graph fast-path. `get_scheduler_state`
//! and `select_sequences_for_phase` are the private helpers used by
//! `build_batch_with_graph` to assemble a batch and decide whether to
//! route it through `GraphBatch::Graph` or `GraphBatch::Regular`.

use std::collections::HashSet;

use vllm_traits::Batch;

use crate::scheduler::cuda_graph::{GraphBatch, GraphPreparedBatch};
use crate::types::{Phase, SeqId, Sequence, Status};

use super::state::SchedulerEngine;

impl SchedulerEngine {
    /// Build batch with potential CUDA Graph routing
    pub fn build_batch_with_graph(&mut self) -> GraphBatch {
        let phase = self
            .phase_scheduler
            .select_phase(&self.get_scheduler_state());
        let (sequences, admitted) = self.select_sequences_for_phase(phase);

        if sequences.is_empty() {
            return GraphBatch::Regular(Batch::empty());
        }

        let batch = self.batch_composer.compose(sequences.clone(), phase);

        // RIL ISS-072 / TASK-085: same overflow requeue as `build_batch`
        // (ISS-041). `compose` serves at most `max_batch_size` / the token
        // budget; a drained-but-uncomposed prefill would otherwise stay
        // `Prefilling` in `running` forever (never re-scheduled) with its
        // pre-allocated KV pinned. The graph path previously skipped this.
        self.requeue_uncomposed_prefills(&admitted, &batch, phase);

        tracing::debug!(
            phase = ?phase,
            sequences_count = sequences.len(),
            batch_seq_ids = ?batch.seq_ids,
            batch_input_tokens_count = batch.input_tokens.len(),
            batch_total_tokens = batch.input_tokens.iter().map(std::vec::Vec::len).sum::<usize>(),
            "build_batch_with_graph: built batch"
        );

        // Only use CUDA Graph for decode phase
        match phase {
            Phase::Prefill => GraphBatch::Regular(batch),
            Phase::Decode => {
                let batch_size = batch.seq_ids.len();
                if self.cuda_graph.enabled && self.cuda_graph.supports_batch_size(batch_size) {
                    self.metrics.record_cuda_graph_hit();
                    GraphBatch::Graph(GraphPreparedBatch::new(batch))
                } else {
                    self.metrics.record_cuda_graph_miss();
                    GraphBatch::Regular(batch)
                }
            }
        }
    }

    /// Get current scheduler state for phase selection
    pub(super) fn get_scheduler_state(&self) -> crate::scheduler::SchedulerState {
        crate::scheduler::SchedulerState {
            waiting_count: self.request_queue.len(),
            running_count: self.running.len(),
            prefill_queue_len: self.request_queue.phase_len(Phase::Prefill),
            decode_queue_len: self.request_queue.phase_len(Phase::Decode),
            available_memory: self.memory.available_blocks(),
            consecutive_decode_rounds: 0,
        }
    }

    /// Select sequences for the given phase — the CUDA-Graph sibling of
    /// [`SchedulerEngine::build_batch`](super::state::SchedulerEngine::build_batch).
    ///
    /// Returns the sequences to compose *and* the ids of the newly-admitted
    /// prefill sequences, so [`Self::build_batch_with_graph`] can re-queue
    /// any that `compose` does not serve (ISS-041). Mirrors `build_batch`'s
    /// guard order so the graph path applies the SAME memory-admission
    /// hardening the regular path got in ISS-042/052/055/056 (RIL ISS-072):
    ///
    /// 1. **ISS-055/056** preemption gate demands only the *additional*
    ///    blocks a sequence still needs — a running decode owning its full
    ///    table must not be re-charged for the whole table (pre-fix
    ///    over-preempted it into a self-livelock under pressure).
    /// 2. **ISS-022/028** pre-allocate the newly-admitted prefills' KV
    ///    blocks before the forward writes them.
    /// 3. **ISS-042** drop any prefill whose table could not be fully
    ///    pre-allocated (re-queue, never forward a short prefill table).
    /// 4. **ISS-052** for a Decode round, grow each running sequence's KV
    ///    table to cover its current tokens (or defer it) so no short-table
    ///    decode is forwarded to the block-0 fallback.
    pub(super) fn select_sequences_for_phase(
        &mut self,
        phase: Phase,
    ) -> (Vec<Sequence>, HashSet<SeqId>) {
        // Only include running decode sequences when in Decode phase —
        // mirrored from `build_batch` (a Prefill round composes only fresh
        // prefills; running `Decoding` sequences are re-included on the next
        // decode round).
        let mut sequences: Vec<Sequence> = if phase == Phase::Decode {
            self.running
                .iter()
                .filter(|s| s.status == Status::Decoding)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        let mut new_sequences = self.request_queue.drain_by_phase(phase);

        // RIL ISS-055/056: memory-pressure admission gate, demanding only
        // the ADDITIONAL blocks each sequence still needs. Pre-fix this
        // compared the full block count, so a running decode holding its
        // whole table re-charged itself for it and got needlessly preempted
        // when a new prefill fit in the free pool — and then re-demanded it
        // again on the next round (self-preemption livelock). Same
        // additional-form as `build_batch` (state/batch.rs).
        for seq in sequences.iter().chain(new_sequences.iter()) {
            let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
            let additional = blocks_needed.saturating_sub(seq.kv_blocks.len());
            if additional > self.memory.available_blocks() {
                self.execute_preemption(additional);
            }
        }

        // RIL ISS-022/028: allocate the newly-admitted sequences' KV blocks
        // before the forward writes their KV — same contract as
        // `build_batch`. The CUDA-Graph builder previously skipped this, so
        // every prefill on `cuda-graph` builds wrote its KV to the block-0
        // fallback and corrupted the cache.
        self.preallocate_kv_blocks(&mut new_sequences);

        // ISS-042: do not admit a prefill whose KV block table could not be
        // fully pre-allocated (defer to a later round instead of serving a
        // short prefill table to the block-0 fallback).
        new_sequences = self.filter_fully_allocated_prefills(new_sequences, phase);

        // RIL ISS-052: for a Decode round, grow each running `Decoding`
        // sequence's KV table to cover its current tokens before
        // composing — or defer any that still cannot obtain its growth
        // block. Pre-fix the graph path cloned short-table decode sequences
        // as-is, forwarding them to the block-0 fallback.
        if phase == Phase::Decode {
            sequences = self.admit_decode_sequences();
        }

        // Add new sequences to the batch (now with allocated KV blocks).
        sequences.extend(new_sequences.iter().cloned());

        // Track which freshly-admitted prefill sequences were moved to
        // running, so `build_batch_with_graph` can re-queue any that
        // `compose` below does not serve.
        let admitted: HashSet<SeqId> = new_sequences.iter().map(|s| s.id).collect();
        self.running.extend(new_sequences);

        (sequences, admitted)
    }
}
