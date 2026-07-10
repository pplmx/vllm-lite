//! `SchedulerEngine::build_batch` + `schedule` — phase selection, batch
//! composition, preemption trigger, and CUDA Graph / observer hooks.

use std::time::Instant;

use vllm_traits::Batch;

use super::SchedulerEngine;
use crate::scheduler::SchedulerState;
use crate::scheduler::observer::ObserverEvent;
use crate::scheduler::policy::SchedulingContext;
use crate::types::{Phase, Status};

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
        let new_sequences = self.request_queue.drain_by_phase(phase);

        // Update metrics: queue depth after draining
        self.metrics
            .set_queue_depth(self.request_queue.len() as u64);

        // If no running decode sequences and no new sequences, return empty
        if sequences.is_empty() && new_sequences.is_empty() {
            return Batch::empty();
        }

        // Add new sequences to the batch
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

        // Check memory and preempt if needed
        for seq in &sequences {
            let blocks_needed = seq.tokens.len().div_ceil(vllm_traits::BLOCK_SIZE);
            if blocks_needed > self.memory.available_blocks() {
                self.execute_preemption(blocks_needed);
            }
        }

        // Move new sequences to running
        self.running.extend(new_sequences);

        // Update metrics: active sequences
        self.metrics.set_active_sequences(self.running.len() as u64);

        // Build the batch
        let batch = self.batch_composer.compose(sequences.clone(), phase);

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
