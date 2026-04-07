use crate::types::SchedulerConfig;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct BatchSnapshot {
    pub timestamp: Instant,
    pub batch_size: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub total_tokens: usize,
    pub latency_ms: f64,
}

#[derive(Clone, Debug)]
pub struct BatchPlan {
    pub target_batch_size: usize,
    pub prefill_budget: usize,
    pub decode_budget: usize,
    pub max_concurrent_prefill: usize,
    pub decode_throughput_hint: f64,
}

pub trait SchedulerStateView {
    fn waiting_count(&self) -> usize;
    fn running_count(&self) -> usize;
    fn prefill_count(&self) -> usize;
    fn decode_count(&self) -> usize;
    fn available_memory(&self) -> usize;
}

pub struct BatchPlanner {
    history: Vec<BatchSnapshot>,
    config: SchedulerConfig,
    max_history_size: usize,
}

impl BatchPlanner {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            history: Vec::with_capacity(100),
            config,
            max_history_size: 100,
        }
    }

    pub fn plan(&mut self, state: &impl SchedulerStateView) -> BatchPlan {
        let adaptive_ratio = self.compute_adaptive_ratio(state);
        let budget = self.config.max_num_batched_tokens;

        BatchPlan {
            target_batch_size: self.predict_optimal_size(state),
            prefill_budget: (budget as f32 * (1.0 - adaptive_ratio)) as usize,
            decode_budget: (budget as f32 * adaptive_ratio) as usize,
            max_concurrent_prefill: self.predict_max_prefill_parallelism(state),
            decode_throughput_hint: self.estimate_throughput(),
        }
    }

    fn compute_adaptive_ratio(&self, state: &impl SchedulerStateView) -> f32 {
        let waiting = state.waiting_count();
        let running = state.running_count();

        if waiting == 0 && running == 0 {
            return 0.7;
        }

        let running_decode = state.decode_count();
        let running_prefill = state.prefill_count();

        if waiting > running && running_decode > running_prefill {
            return 0.4;
        }

        if running_prefill > running_decode.saturating_mul(2) {
            return 0.8;
        }

        0.7
    }

    fn predict_optimal_size(&self, state: &impl SchedulerStateView) -> usize {
        let waiting = state.waiting_count();
        let running = state.running_count();
        let available = state.available_memory();

        let base = self.config.max_num_seqs;

        let memory_factor = if available < 10 {
            0.5
        } else if available < 50 {
            0.7
        } else if available > 500 {
            1.2
        } else {
            1.0
        };

        let queue_factor = if waiting > 100 {
            1.3
        } else if waiting > 50 {
            1.1
        } else {
            1.0
        };

        let result = (base as f32 * memory_factor * queue_factor) as usize;
        result
            .min(waiting + running)
            .max(self.config.min_batch_size)
    }

    fn predict_max_prefill_parallelism(&self, state: &impl SchedulerStateView) -> usize {
        let memory = state.available_memory();
        let running = state.running_count();

        let memory_based = memory / 16;

        let running_based = running.saturating_sub(state.decode_count());

        memory_based
            .min(running_based)
            .max(1)
            .min(self.config.max_num_seqs / 2)
    }

    fn estimate_throughput(&self) -> f64 {
        if self.history.is_empty() {
            return 1000.0;
        }

        let valid: Vec<_> = self
            .history
            .iter()
            .rev()
            .take(10)
            .filter(|s| s.latency_ms > 0.0)
            .collect();

        if valid.is_empty() {
            return 1000.0;
        }

        let sum: f64 = valid
            .iter()
            .map(|s| s.total_tokens as f64 / s.latency_ms * 1000.0)
            .sum();

        sum / valid.len() as f64
    }

    pub fn record(&mut self, snapshot: BatchSnapshot) {
        if self.history.len() >= self.max_history_size {
            self.history.remove(0);
        }
        self.history.push(snapshot);
    }

    pub fn get_stats(&self) -> (f64, f64, f64) {
        if self.history.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let total_batch: f64 = self
            .history
            .iter()
            .map(|s| s.batch_size as f64)
            .sum::<f64>()
            / self.history.len() as f64;
        let total_latency: f64 =
            self.history.iter().map(|s| s.latency_ms).sum::<f64>() / self.history.len() as f64;
        let throughput = self.estimate_throughput();

        (total_batch, total_latency, throughput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockState {
        waiting: usize,
        running: usize,
        prefill: usize,
        decode: usize,
        memory: usize,
    }

    impl SchedulerStateView for MockState {
        fn waiting_count(&self) -> usize {
            self.waiting
        }
        fn running_count(&self) -> usize {
            self.running
        }
        fn prefill_count(&self) -> usize {
            self.prefill
        }
        fn decode_count(&self) -> usize {
            self.decode
        }
        fn available_memory(&self) -> usize {
            self.memory
        }
    }

    #[test]
    fn test_batch_plan_creation() {
        let config = SchedulerConfig::default();
        let mut planner = BatchPlanner::new(config);

        let state = MockState {
            waiting: 10,
            running: 5,
            prefill: 2,
            decode: 3,
            memory: 100,
        };

        let plan = planner.plan(&state);
        assert!(plan.target_batch_size > 0);
        assert!(plan.prefill_budget > 0);
        assert!(plan.decode_budget > 0);
    }

    #[test]
    fn test_adaptive_ratio_favors_prefill_when_waiting() {
        let config = SchedulerConfig::default();
        let mut planner = BatchPlanner::new(config);

        let state = MockState {
            waiting: 50,
            running: 5,
            prefill: 1,
            decode: 4,
            memory: 100,
        };

        let plan = planner.plan(&state);
        assert!(plan.prefill_budget > plan.decode_budget);
    }

    #[test]
    fn test_adaptive_ratio_favors_decode_when_running() {
        let config = SchedulerConfig::default();
        let mut planner = BatchPlanner::new(config);

        let state = MockState {
            waiting: 0,
            running: 10,
            prefill: 8,
            decode: 2,
            memory: 100,
        };

        let plan = planner.plan(&state);
        assert!(plan.decode_budget > plan.prefill_budget);
    }
}
