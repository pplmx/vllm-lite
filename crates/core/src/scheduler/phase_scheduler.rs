use crate::types::Phase;

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

#[derive(Clone, Debug)]
pub struct SchedulerState {
    pub waiting_count: usize,
    pub running_count: usize,
    pub prefill_queue_len: usize,
    pub decode_queue_len: usize,
    pub available_memory: usize,
    pub consecutive_decode_rounds: u32,
}

pub struct PhaseScheduler {
    current_phase: Phase,
    switch_policy: PhaseSwitchPolicy,
    consecutive_decode_rounds: u32,
}

impl PhaseScheduler {
    pub fn new(switch_policy: PhaseSwitchPolicy) -> Self {
        Self {
            current_phase: Phase::Prefill,
            switch_policy,
            consecutive_decode_rounds: 0,
        }
    }

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
                if self.prefill_complete(state) {
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

    pub fn current_phase(&self) -> Phase {
        self.current_phase
    }

    pub fn reset(&mut self) {
        self.current_phase = Phase::Prefill;
        self.consecutive_decode_rounds = 0;
    }

    fn should_switch_to_prefill(&self, state: &SchedulerState) -> bool {
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

    fn prefill_complete(&self, state: &SchedulerState) -> bool {
        state.prefill_queue_len == 0
    }
}

impl Default for PhaseScheduler {
    fn default() -> Self {
        Self::new(PhaseSwitchPolicy::default())
    }
}

#[cfg(test)]
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
