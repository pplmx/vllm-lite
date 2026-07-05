#![allow(clippy::module_name_repetitions)]
//! `TestHarness` - Common test environment setup
//!
//! Provides a unified interface for initializing test environments
//! with common utilities and configurations.

use std::sync::Arc;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::scheduler::SchedulerEngine;
use vllm_core::types::{Request, SchedulerConfig, SequencePackingConfig};

/// Test harness configuration
#[derive(Debug, Clone)]
pub struct TestHarnessConfig {
    /// Number of KV-cache blocks to allocate for the test scheduler.
    pub kv_blocks: usize,
    /// Maximum batch size the test scheduler may produce.
    pub max_batch_size: usize,
    /// Whether to enable prefix-cache sharing across requests.
    pub enable_prefix_cache: bool,
    /// Whether to enable dynamic batching heuristics.
    pub enable_dynamic_batching: bool,
}

impl Default for TestHarnessConfig {
    fn default() -> Self {
        Self {
            kv_blocks: 256,
            max_batch_size: 16,
            enable_prefix_cache: true,
            enable_dynamic_batching: false,
        }
    }
}

impl TestHarnessConfig {
    #[must_use]
    pub const fn kv_blocks(mut self, n: usize) -> Self {
        self.kv_blocks = n;
        self
    }

    #[must_use]
    pub const fn max_batch_size(mut self, n: usize) -> Self {
        self.max_batch_size = n;
        self
    }

    #[must_use]
    pub const fn enable_prefix_cache(mut self, enabled: bool) -> Self {
        self.enable_prefix_cache = enabled;
        self
    }

    #[must_use]
    pub const fn enable_dynamic_batching(mut self, enabled: bool) -> Self {
        self.enable_dynamic_batching = enabled;
        self
    }

    #[must_use]
    pub fn into_scheduler_config(self) -> SchedulerConfig {
        SchedulerConfig::new(
            self.max_batch_size,
            4096,
            10,
            true,
            512,
            0.7,
            false,
            self.enable_dynamic_batching,
            1,
            self.max_batch_size,
            SequencePackingConfig::default(),
        )
    }
}

/// Test harness for vllm-lite integration tests
///
/// Provides a unified setup for tests with common utilities:
/// - `SchedulerEngine` with configurable settings
/// - Metrics collector for tracking
/// - Mock model support
///
/// # Example
///
/// ```rust,ignore
/// use vllm_testing::TestHarness;
///
/// let harness = TestHarness::new()
///     .kv_blocks(128)
///     .max_batch_size(8)
///     .build();
///
/// let mut scheduler = harness.scheduler();
/// let seq_id = scheduler.add_request(Request::new(0, vec![1, 2, 3], 10));
/// ```
#[derive(Debug)]
pub struct TestHarness {
    /// Harness configuration (mirrors into the scheduler).
    pub config: TestHarnessConfig,
    /// Shared metrics collector for the test process.
    pub metrics: Arc<EnhancedMetricsCollector>,
    /// Pre-built scheduler engine wired to the metrics collector.
    pub scheduler: SchedulerEngine,
}

impl TestHarness {
    /// Create a new `TestHarness` with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(TestHarnessConfig::default())
    }

    /// Create a `TestHarness` from a custom configuration
    #[must_use]
    pub fn from_config(config: TestHarnessConfig) -> Self {
        let metrics = Arc::new(EnhancedMetricsCollector::new());
        let scheduler_config = config.clone().into_scheduler_config();

        let scheduler = SchedulerEngine::new(scheduler_config, config.kv_blocks, metrics.clone());

        Self {
            config,
            metrics,
            scheduler,
        }
    }

    /// Get a reference to the scheduler engine
    pub const fn scheduler(&self) -> &SchedulerEngine {
        &self.scheduler
    }

    /// Get a mutable reference to the scheduler engine
    pub const fn scheduler_mut(&mut self) -> &mut SchedulerEngine {
        &mut self.scheduler
    }

    /// Get a reference to the metrics collector
    pub const fn metrics(&self) -> &Arc<EnhancedMetricsCollector> {
        &self.metrics
    }

    /// Reset the metrics collector
    pub fn reset_metrics(&self) {
        let _ = self.metrics.get_counter("requests_total");
    }

    /// Add a test request to the scheduler
    pub fn add_test_request(&mut self, prompt: Vec<u32>, max_tokens: usize) -> u64 {
        let request = Request::new(0, prompt, max_tokens);
        self.scheduler.add_request(request)
    }

    /// Build a test batch from waiting requests
    pub fn build_test_batch(&mut self) -> vllm_traits::Batch {
        self.scheduler.build_batch()
    }

    /// Check if there are pending requests
    pub fn has_pending(&self) -> bool {
        self.scheduler.has_pending()
    }

    /// Get the number of waiting requests
    pub fn waiting_count(&self) -> usize {
        self.scheduler.waiting_count()
    }

    /// Get the number of running requests
    pub const fn running_count(&self) -> usize {
        self.scheduler.running_count()
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_default_config() {
        let harness = TestHarness::new();
        assert_eq!(harness.config.kv_blocks, 256);
        assert_eq!(harness.config.max_batch_size, 16);
    }

    #[test]
    fn test_harness_custom_config() {
        let harness = TestHarness::from_config(
            TestHarnessConfig::default()
                .kv_blocks(128)
                .max_batch_size(8),
        );
        assert_eq!(harness.config.kv_blocks, 128);
        assert_eq!(harness.config.max_batch_size, 8);
    }

    #[test]
    fn test_harness_add_request() {
        let mut harness = TestHarness::new();
        let seq_id = harness.add_test_request(vec![1, 2, 3], 10);
        assert!(seq_id > 0);
        assert_eq!(harness.waiting_count(), 1);
    }

    #[test]
    fn test_harness_build_batch() {
        let mut harness = TestHarness::new();
        harness.add_test_request(vec![1, 2, 3], 10);

        let batch = harness.build_test_batch();
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_harness_has_pending() {
        let mut harness = TestHarness::new();
        assert!(!harness.has_pending());

        harness.add_test_request(vec![1, 2, 3], 10);
        assert!(harness.has_pending());
    }
}
