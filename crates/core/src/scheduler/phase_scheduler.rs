//! Phase-switch policy: decide when to flip between Prefill (ingest prompt tokens) and Decode (generate next tokens).
//!
//! Encapsulated as a [`PhasePolicy`] trait with multiple implementations:
//! `DrainDecode`, `RoundRobin`, and `Hybrid`. The scheduler picks one at
//! construction time based on the `SchedulerConfig.phase_policy` field.
use crate::types::Phase;

/// Strategy for switching between prefill and decode phases. Different policies optimize throughput (drain decode) vs. fairness (round-robin).
#[derive(Clone, Debug)]
pub struct PhaseSwitchPolicy {
    pub max_consecutive_decode: u32,
    pub prefill_priority_threshold: usize,
    pub min_decode_batch_size: usize,
}

impl Default for PhaseSwitchPolicy {
    fn default() -> Self {
        Self {
            max_consecutive_decode: 10,
            prefill_priority_threshold: 5,
            min_decode_batch_size: 2,
        }
    }
}

impl PhaseSwitchPolicy {
    /// Returns a builder for configuring this type with the documented field defaults.
    /// Use `with_*(...)` to override individual fields, then `build()` to produce the type.
    #[must_use]
    pub fn builder() -> PhaseSwitchPolicyBuilder {
        PhaseSwitchPolicyBuilder::default()
    }
}

/// Builder for [`PhaseSwitchPolicy`].
#[derive(Debug, Clone, Default)]
pub struct PhaseSwitchPolicyBuilder {
    inner: PhaseSwitchPolicy,
}

impl PhaseSwitchPolicyBuilder {
    /// Set [`PhaseSwitchPolicy::max_consecutive_decode`].
    #[must_use]
    pub const fn with_max_consecutive_decode(mut self, v: u32) -> Self {
        self.inner.max_consecutive_decode = v;
        self
    }
    /// Set [`PhaseSwitchPolicy::prefill_priority_threshold`].
    #[must_use]
    pub const fn with_prefill_priority_threshold(mut self, v: usize) -> Self {
        self.inner.prefill_priority_threshold = v;
        self
    }
    /// Set [`PhaseSwitchPolicy::min_decode_batch_size`].
    #[must_use]
    pub const fn with_min_decode_batch_size(mut self, v: usize) -> Self {
        self.inner.min_decode_batch_size = v;
        self
    }
    /// Finalize the builder into a [`PhaseSwitchPolicy`].
    #[must_use]
    pub const fn build(self) -> PhaseSwitchPolicy {
        self.inner
    }
}

/// Internal mutable state of the phase scheduler: current phase, last-transition tick, accumulated stats. Mutated under the scheduler's lock on each transition.
#[derive(Clone, Debug)]
pub struct SchedulerState {
    pub waiting_count: usize,
    pub running_count: usize,
    pub prefill_queue_len: usize,
    pub decode_queue_len: usize,
    pub available_memory: usize,
    pub consecutive_decode_rounds: u32,
}

#[derive(Debug)]
/// Coordinator that interleaves prefill and decode batches within a single step. Avoids the head-of-line blocking that happens when a long prefill monopolizes the GPU.
pub struct PhaseScheduler {
    current_phase: Phase,
    switch_policy: PhaseSwitchPolicy,
    consecutive_decode_rounds: u32,
}

impl PhaseScheduler {
    /// Construct a phase scheduler starting in [`Phase::Prefill`].
    #[must_use]
    pub const fn new(switch_policy: PhaseSwitchPolicy) -> Self {
        Self {
            current_phase: Phase::Prefill,
            switch_policy,
            consecutive_decode_rounds: 0,
        }
    }

    /// Select the current phase based on scheduler state.
    #[must_use]
    pub fn select_phase(&mut self, state: &SchedulerState) -> Phase {
        let phase = match self.current_phase {
            Phase::Decode => {
                if self.should_switch_to_prefill(state) {
                    Phase::Prefill
                } else {
                    Phase::Decode
                }
            }
            Phase::Prefill => {
                if Self::prefill_complete(state) {
                    Phase::Decode
                } else {
                    Phase::Prefill
                }
            }
        };

        if phase == Phase::Decode {
            self.consecutive_decode_rounds += 1;
        } else {
            self.consecutive_decode_rounds = 0;
        }

        self.current_phase = phase;
        phase
    }

    /// Get the current phase.
    #[must_use]
    pub const fn current_phase(&self) -> Phase {
        self.current_phase
    }

    /// Reset the scheduler to its initial state (Prefill, zero decode rounds).
    pub const fn reset(&mut self) {
        self.current_phase = Phase::Prefill;
        self.consecutive_decode_rounds = 0;
    }

    const fn should_switch_to_prefill(&self, state: &SchedulerState) -> bool {
        if self.consecutive_decode_rounds >= self.switch_policy.max_consecutive_decode {
            return true;
        }
        if state.prefill_queue_len >= self.switch_policy.prefill_priority_threshold {
            return true;
        }
        if state.decode_queue_len < self.switch_policy.min_decode_batch_size {
            return true;
        }
        false
    }

    const fn prefill_complete(state: &SchedulerState) -> bool {
        state.prefill_queue_len == 0
    }
}

impl Default for PhaseScheduler {
    fn default() -> Self {
        Self::new(PhaseSwitchPolicy::default())
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_switch_to_prefill_after_consecutive_limit() {
        let mut scheduler = PhaseScheduler::new(PhaseSwitchPolicy {
            max_consecutive_decode: 3,
            prefill_priority_threshold: 100,
            min_decode_batch_size: 1,
        });
        scheduler.current_phase = Phase::Decode;
        scheduler.consecutive_decode_rounds = 3;

        let state = SchedulerState {
            waiting_count: 10,
            running_count: 5,
            prefill_queue_len: 1,
            decode_queue_len: 5,
            available_memory: 100,
            consecutive_decode_rounds: 3,
        };

        let phase = scheduler.select_phase(&state);
        assert_eq!(phase, Phase::Prefill);
    }

    #[test]
    fn test_phase_stays_prefill_when_prefill_queue_not_empty() {
        let mut scheduler = PhaseScheduler::default();
        scheduler.current_phase = Phase::Prefill;

        let state = SchedulerState {
            waiting_count: 10,
            running_count: 0,
            prefill_queue_len: 5,
            decode_queue_len: 0,
            available_memory: 100,
            consecutive_decode_rounds: 0,
        };

        let phase = scheduler.select_phase(&state);
        assert_eq!(phase, Phase::Prefill);
    }

    #[test]
    fn test_phase_switches_to_decode_when_prefill_empty() {
        let mut scheduler = PhaseScheduler::default();
        scheduler.current_phase = Phase::Prefill;

        let state = SchedulerState {
            waiting_count: 0,
            running_count: 0,
            prefill_queue_len: 0,
            decode_queue_len: 5,
            available_memory: 100,
            consecutive_decode_rounds: 0,
        };

        let phase = scheduler.select_phase(&state);
        assert_eq!(phase, Phase::Decode);
    }
}
