//! Test fixtures for common scenarios.

use vllm_core::scheduler::cuda_graph::SchedulerCudaGraphConfig;
use vllm_core::types::{SchedulerConfig, SequencePackingConfig};

pub struct TestFixtures;

impl TestFixtures {
    pub fn default_scheduler_config() -> SchedulerConfig {
        SchedulerConfig::default()
    }

    pub fn small_batch_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 2,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing: SequencePackingConfig::default(),
        }
    }

    pub fn chunked_prefill_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 10,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing: SequencePackingConfig::default(),
        }
    }

    pub fn pd_separation_config() -> SchedulerConfig {
        SchedulerConfig {
            enable_pd_separation: true,
            decode_preference_ratio: 0.5,
            max_num_seqs: 10,
            max_num_batched_tokens: 100,
            max_consecutive_decode: 10,
            prefill_chunk_size: 512,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 256,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing: SequencePackingConfig::default(),
        }
    }

    pub fn priority_config() -> SchedulerConfig {
        SchedulerConfig {
            enable_priority_scheduling: true,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            ..SchedulerConfig::default()
        }
    }

    pub fn oom_scenario_config() -> SchedulerConfig {
        SchedulerConfig {
            max_num_seqs: 1,
            max_num_batched_tokens: 1,
            max_consecutive_decode: 10,
            enable_pd_separation: false,
            prefill_chunk_size: 512,
            decode_preference_ratio: 0.7,
            enable_priority_scheduling: false,
            enable_dynamic_batching: false,
            min_batch_size: 1,
            max_batch_size: 2,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            packing: SequencePackingConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TestFixtures::default_scheduler_config();
        assert_eq!(config.max_num_seqs, 256);
    }

    #[test]
    fn test_small_batch_config() {
        let config = TestFixtures::small_batch_config();
        assert_eq!(config.max_num_seqs, 2);
    }

    #[test]
    fn test_oom_scenario_config() {
        let config = TestFixtures::oom_scenario_config();
        assert_eq!(config.max_num_seqs, 1);
        assert_eq!(config.max_num_batched_tokens, 1);
    }
}
