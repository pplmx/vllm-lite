//! Preemption: when KV-cache memory is tight, pick a running sequence to evict and re-queue it.
//!
//! `PreemptionManager` owns the eviction policy (LRU by default, with
//! priority-weighted overrides). `select_victim`, `record_preemption`,
//! `record_rejection`, and `reset_stats` form the public surface that
//! the scheduler engine wires into its tick loop.
//! `select_victim`, `record_preemption`, `record_rejection`, and
// `reset_stats` are public API surface on `PreemptionManager`. The
// scheduler engine wires them when it adopts the preemption-driven
// admission path; current uses bypass them.
#![allow(dead_code)]

use crate::types::{SchedulerConfig, Sequence, Status};

/// Implements the preemption policy: when a new request cannot fit, decides which running sequence to preempt (LRU, priority, longest-job-first).
#[derive(Debug)]
pub(crate) struct PreemptionManager {
    config: SchedulerConfig,
    preempted_count: u64,
    rejected_count: u64,
}

impl Default for PreemptionManager {
    fn default() -> Self {
        Self::new(SchedulerConfig::default())
    }
}

impl PreemptionManager {
    pub const fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            preempted_count: 0,
            rejected_count: 0,
        }
    }

    pub fn should_preempt(
        &self,
        running_len: usize,
        waiting_len: usize,
        blocks_needed: usize,
        blocks_available: usize,
    ) -> bool {
        tracing::debug!(
            running = running_len,
            waiting = waiting_len,
            blocks_needed,
            blocks_available,
            "Preemption check"
        );

        if waiting_len == 0 || running_len == 0 {
            return false;
        }

        if blocks_available >= blocks_needed {
            return false;
        }

        if running_len <= self.config.min_batch_size.max(1) {
            return false;
        }

        if blocks_available <= 1 {
            return false;
        }

        // Scale aggressiveness with configured batch pressure.
        // invariant: max_num_seqs/max_batch_size are bounded config values; f32
        // precision loss is acceptable for the pressure ratio.
        #[allow(clippy::cast_precision_loss)]
        let shortage_threshold = (self.config.max_num_seqs as f32
            / self.config.max_batch_size as f32)
            .min(2.0)
            .mul_add(0.1, 1.0);
        // invariant: block counts are bounded by available memory, far below 2^24.
        #[allow(clippy::cast_precision_loss)]
        let memory_shortage_ratio = blocks_needed as f32 / (blocks_available - 1) as f32;
        if memory_shortage_ratio < shortage_threshold {
            return false;
        }

        true
    }

    #[allow(dead_code)] // test-only helper; reachable under cfg(test) only
    pub(crate) fn select_victim(&self, running: &[Sequence]) -> Option<(usize, Sequence)> {
        tracing::debug!(candidates = running.len(), "Selecting preemption victim");

        if running.len() <= self.config.min_batch_size.max(1) {
            return None;
        }

        let decode_sequences: Vec<_> = running
            .iter()
            .enumerate()
            .filter(|(_, s)| s.status == Status::Decoding)
            .collect();

        let victim = if decode_sequences.is_empty() {
            running
                .iter()
                .enumerate()
                .min_by_key(|(_, seq)| seq.consecutive_decode_rounds)
                .map(|(idx, seq)| (idx, seq.clone()))
        } else {
            decode_sequences
                .into_iter()
                .min_by_key(|(_, seq)| seq.consecutive_decode_rounds)
                .map(|(idx, seq)| (idx, seq.clone()))
        };

        if let Some((idx, ref seq)) = victim {
            tracing::trace!(seq_id = seq.id, idx, "Preemption victim selected");
        }

        victim
    }

    pub const fn preempted_count(&self) -> u64 {
        self.preempted_count
    }

    pub const fn rejected_count(&self) -> u64 {
        self.rejected_count
    }

    pub const fn record_preemption(&mut self) {
        self.preempted_count += 1;
    }

    pub const fn record_rejection(&mut self) {
        self.rejected_count += 1;
    }

    pub const fn reset_stats(&mut self) {
        self.preempted_count = 0;
        self.rejected_count = 0;
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// preemption module under the 800-line soft cap. They cover the
// four-input `should_preempt` decision, the least-progress-first
// victim selector, and the preempted/rejected stat counters.
#[cfg(test)]
mod tests;
