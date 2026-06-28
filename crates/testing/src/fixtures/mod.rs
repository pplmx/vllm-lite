#![allow(clippy::module_name_repetitions)]
//! Test fixtures for common scenarios.

use vllm_core::engine::Engine;
use vllm_core::scheduler::cuda_graph::SchedulerCudaGraphConfig;
use vllm_core::types::{SchedulerConfig, SequencePackingConfig};

use crate::mocks::IncrementModel;

#[derive(Debug)]
/// `TestFixtures`: test fixtures.
pub struct TestFixtures;

impl TestFixtures {
    #[must_use]
    pub fn default_scheduler_config() -> SchedulerConfig {
        SchedulerConfig::default()
    }

    #[must_use]
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

    #[must_use]
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

    #[must_use]
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

    #[must_use]
    pub fn priority_config() -> SchedulerConfig {
        SchedulerConfig {
            enable_priority_scheduling: true,
            cuda_graph: SchedulerCudaGraphConfig::default(),
            ..SchedulerConfig::default()
        }
    }

    #[must_use]
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

    /// Engine backed by [`IncrementModel`] with the default scheduler config.
    #[must_use]
    pub fn increment_engine(kv_blocks: usize) -> Engine {
        Self::increment_engine_with(Self::default_scheduler_config(), 4, kv_blocks)
    }

    /// Engine backed by [`IncrementModel`] with a custom scheduler config.
    #[must_use]
    pub fn increment_engine_with(
        config: SchedulerConfig,
        max_draft_tokens: usize,
        kv_blocks: usize,
    ) -> Engine {
        Engine::with_config(IncrementModel, None, config, max_draft_tokens, kv_blocks)
    }

    /// Target + draft engine for speculative-decoding E2E tests.
    #[must_use]
    pub fn increment_speculative_engine(kv_blocks: usize) -> Engine {
        Self::increment_speculative_engine_with(Self::default_scheduler_config(), 4, kv_blocks)
    }

    /// Speculative engine with a custom scheduler config.
    #[must_use]
    pub fn increment_speculative_engine_with(
        config: SchedulerConfig,
        max_draft_tokens: usize,
        kv_blocks: usize,
    ) -> Engine {
        Engine::with_config(
            IncrementModel,
            Some(IncrementModel),
            config,
            max_draft_tokens,
            kv_blocks,
        )
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
