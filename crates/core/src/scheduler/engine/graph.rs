//! Graph helper methods for `SchedulerEngine`.
//!
//! These methods drive the CUDA Graph fast-path. `get_scheduler_state`
//! and `select_sequences_for_phase` are the private helpers used by
//! `build_batch_with_graph` to assemble a batch and decide whether to
//! route it through `GraphBatch::Graph` or `GraphBatch::Regular`.

use vllm_traits::Batch;

use crate::scheduler::cuda_graph::{GraphBatch, GraphPreparedBatch};
use crate::types::{Phase, Sequence, Status};

use super::state::SchedulerEngine;

impl SchedulerEngine {
    /// Build batch with potential CUDA Graph routing
    pub fn build_batch_with_graph(&mut self) -> GraphBatch {
        let phase = self
            .phase_scheduler
            .select_phase(&self.get_scheduler_state());
        let sequences = self.select_sequences_for_phase(phase);

        if sequences.is_empty() {
            return GraphBatch::Regular(Batch::empty());
        }

        let batch = self.batch_composer.compose(sequences.clone(), phase);

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

    /// Select sequences for the given phase
    pub(super) fn select_sequences_for_phase(&mut self, phase: Phase) -> Vec<Sequence> {
        let mut sequences: Vec<Sequence> = self
            .running
            .iter()
            .filter(|s| s.status == Status::Decoding)
            .cloned()
            .collect();

        let new_sequences = self.request_queue.drain_by_phase(phase);
        sequences.extend(new_sequences.iter().cloned());
        self.running.extend(new_sequences);

        sequences
    }
}
