//! SchedulerEngine — continuous-batching engine.
//!
//! This module is split into four focused sub-modules that share a
//! single `SchedulerEngine` struct (defined in [`state`]):
//!
//! - [`graph`] (3 methods) — CUDA Graph helpers (`build_batch_with_graph`
//!   plus the two private helpers it relies on).
//! - [`update`] — `update`, the post-step state update invoked after
//!   the model forward pass returns.
//! - [`memory`] (6 methods) — preemption, KV cache rollback, request
//!   cancellation, pressure reporting, KV usage stats, and the
//!   prefix cache accessor.
//! - [`state`] — the struct itself, its constructor, the major
//!   lifecycle methods (`add_request`, `build_batch`, `schedule`,
//!   `set_policy`), the `Default` impl, and the read-only state
//!   accessors.
//!
//! All `impl SchedulerEngine` blocks across the sub-modules contribute
//! methods to the same type, so callers see a single flat API.

mod graph;
mod memory;
mod state;
mod update;

pub use graph::*;
pub use memory::*;
pub use state::*;
pub use update::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_traits::BatchPhase;

    use crate::metrics::EnhancedMetricsCollector;
    use crate::scheduler::engine::SchedulerEngine;
    use crate::types::{Request, SchedulerConfig};

    fn create_test_engine(config: SchedulerConfig, num_kv_blocks: usize) -> SchedulerEngine {
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        SchedulerEngine::new(config, num_kv_blocks, metrics)
    }

    #[test]
    fn test_engine_add_request() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        assert!(id > 0);
        assert!(engine.has_pending());
        assert_eq!(engine.waiting_count(), 1);
    }

    #[test]
    fn test_engine_build_batch() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_engine_batch_phase_is_prefill() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let batch = engine.build_batch();
        assert_eq!(batch.phase, BatchPhase::Prefill);
    }

    #[test]
    fn test_engine_update_sequence() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);
        let id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));
        let _batch = engine.build_batch();

        // Simulate model output: one token generated
        engine.update(&[id], &[100], &[3]); // 3 input tokens processed

        // The sequence should still be in running (not finished yet)
        assert_eq!(engine.running_count(), 1);
    }

    #[test]
    fn test_engine_multiple_requests() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);

        // Add multiple requests
        let id1 = engine.add_request(Request::new(0, vec![1, 2], 5));
        let id2 = engine.add_request(Request::new(0, vec![3, 4], 5));

        assert_eq!(engine.waiting_count(), 2);

        let batch = engine.build_batch();
        assert_eq!(batch.seq_ids.len(), 2);
        assert!(batch.seq_ids.contains(&id1));
        assert!(batch.seq_ids.contains(&id2));
    }

    #[test]
    fn test_engine_memory_pressure() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 10); // Small memory

        // Memory pressure should be 0.0 with all blocks free
        assert_eq!(engine.get_memory_pressure(), 0.0);

        // Add a request
        engine.add_request(Request::new(0, vec![1, 2, 3, 4, 5], 5));

        // After building batch, memory pressure may increase
        let _batch = engine.build_batch();

        // Pressure should be between 0 and 1
        let pressure = engine.get_memory_pressure();
        assert!((0.0..=1.0).contains(&pressure));
    }

    #[test]
    fn test_engine_prefix_cache_hit() {
        let config = SchedulerConfig::default();
        let mut engine = create_test_engine(config, 1024);

        // Add first request
        let prompt = vec![1, 2, 3, 4, 5];
        let id1 = engine.add_request(Request::new(0, prompt.clone(), 5));

        // Build batch and process
        let _batch = engine.build_batch();
        engine.update(&[id1], &[100], &[5]);

        // Complete the sequence to add to cache
        // Update until max_tokens reached
        for i in 0..5 {
            engine.update(&[id1], &[100 + i as u32], &[0]);
        }

        // Add second request with same prefix
        let _id2 = engine.add_request(Request::new(0, vec![1, 2, 3, 6, 7], 5));

        // Second request should be enqueued
        assert!(engine.waiting_count() > 0 || engine.running_count() > 0);
    }

    #[test]
    fn test_engine_metrics_tracking() {
        let config = SchedulerConfig::default();
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        let mut engine = SchedulerEngine::new(config, 1024, metrics.clone());

        // Initially metrics should be zero
        assert_eq!(metrics.get_counter("requests_total"), 0);

        // Add a request
        let _id = engine.add_request(Request::new(0, vec![1, 2, 3], 5));

        // Check metrics were updated
        assert_eq!(metrics.get_counter("requests_total"), 1);

        // Build batch to trigger latency recording
        let _batch = engine.build_batch();

        // Metrics should still track request count
        assert_eq!(metrics.get_counter("requests_total"), 1);
    }
}
