// `select_victim`, `record_preemption`, `record_rejection`, and
// `reset_stats` are public API surface on `PreemptionManager`. The
// scheduler engine wires them when it adopts the preemption-driven
// admission path; current uses bypass them.
#![allow(dead_code)]

use crate::types::{SchedulerConfig, Sequence, Status};

/// `PreemptionManager`: preemption manager.
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

    pub fn select_victim(&self, running: &[Sequence]) -> Option<(usize, Sequence)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Priority, SamplingParams, Status};

    fn make_sequence(id: u64, decode_rounds: u32) -> Sequence {
        Sequence {
            id,
            tokens: vec![1, 2, 3],
            kv_blocks: std::sync::Arc::new(vec![]),
            num_computed_tokens: 3,
            prompt_len: 3,
            status: Status::Decoding,
            max_tokens: 100,
            sampling_params: SamplingParams::default(),
            consecutive_decode_rounds: decode_rounds,
            priority: Priority::default(),
            degraded_draft: false,
            draft_model_id: None,
        }
    }

    #[test]
    fn test_should_preempt_no_waiting() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        assert!(!manager.should_preempt(5, 0, 10, 1));
    }

    #[test]
    fn test_should_preempt_no_running() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        assert!(!manager.should_preempt(0, 5, 10, 1));
    }

    #[test]
    fn test_should_preempt_enough_blocks() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        assert!(!manager.should_preempt(5, 3, 10, 15));
    }

    #[test]
    fn test_should_preempt_single_running() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        assert!(!manager.should_preempt(1, 5, 10, 1));
    }

    #[test]
    fn test_should_preempt_all_conditions_met() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        assert!(manager.should_preempt(3, 5, 10, 5));
    }

    #[test]
    fn test_select_victim_single() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        let running = vec![make_sequence(1, 5)];
        assert!(manager.select_victim(&running).is_none());
    }

    #[test]
    fn test_select_victim_multiple() {
        let manager = PreemptionManager::new(SchedulerConfig::default());
        let running = vec![
            make_sequence(1, 10),
            make_sequence(2, 5),
            make_sequence(3, 15),
        ];
        let result = manager.select_victim(&running);
        assert!(result.is_some());
        let (idx, seq) = result.unwrap();
        assert_eq!(idx, 1);
        assert_eq!(seq.id, 2);
    }

    #[test]
    fn test_stats() {
        let mut manager = PreemptionManager::new(SchedulerConfig::default());
        assert_eq!(manager.preempted_count(), 0);
        assert_eq!(manager.rejected_count(), 0);

        manager.record_preemption();
        assert_eq!(manager.preempted_count(), 1);

        manager.record_rejection();
        assert_eq!(manager.rejected_count(), 1);

        manager.reset_stats();
        assert_eq!(manager.preempted_count(), 0);
        assert_eq!(manager.rejected_count(), 0);
    }
}
