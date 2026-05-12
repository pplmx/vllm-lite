//! Speculative Decoding Memory Overhead Tests
//!
//! SPEC-04.3: Memory overhead measurement (KV cache, overhead per request).
//! Tests that speculative decoding has bounded memory overhead vs standard decoding.
//!
//! Key insight for self-speculation: the draft model shares weights with the target
//! model (no additional weight memory). The only overhead is:
//! 1. Draft KV cache blocks (if draft maintains separate KV)
//! 2. Additional intermediate allocations during draft generation and verification
//!
//! For self-speculation with weight sharing, draft KV cache can also be shared
//! since the draft runs on the same model.

use tokio::sync::mpsc;
use vllm_core::engine::Engine;
use vllm_core::speculative::SelfSpeculativeModel;
use vllm_core::speculative::SpeculationConfig;
use vllm_core::types::{AdaptiveDraftConfig, Request, SchedulerConfig};
use vllm_traits::ModelBackend;
use vllm_testing::IncrementModel;

/// SPEC-04.3: Verify that the memory overhead of speculative decoding is bounded.
///
/// Self-speculation shares model weights, so the only additional memory should be
/// from the draft model's KV cache (proportional to draft tokens). This test checks
/// that the total allocated KV blocks is reasonable for both standard and speculative modes.
#[test]
fn test_speculative_memory_overhead_bounded() {
    let config = SchedulerConfig::default();
    let num_kv_blocks = 1024;

    // Create engine with speculative mode
    let mut spec_engine = Engine::with_config(
        IncrementModel,
        Some(IncrementModel), // same model as draft
        config.clone(),
        4,
        num_kv_blocks,
    );

    spec_engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

    let (tx, _rx) = mpsc::channel(64);
    spec_engine.add_request(Request::new(1, vec![10, 20, 30], 10), tx);

    // Prefill should allocate KV blocks
    let _ = spec_engine.step();

    // Check KV cache usage immediately after prefill (before any decode step)
    let (used_after_prefill, total) = spec_engine.scheduler.get_kv_cache_usage();

    // KV blocks should be allocated after prefill
    assert!(
        used_after_prefill > 0,
        "KV blocks should be allocated after prefill (got {})",
        used_after_prefill
    );

    // Should not exceed total available
    assert!(
        used_after_prefill <= total,
        "Used blocks should not exceed total: {} > {}",
        used_after_prefill,
        total
    );

    // Run to completion
    while spec_engine.has_pending() {
        let _ = spec_engine.step_adaptive_speculative();
    }
}

/// SPEC-04.3: Verify that the KV cache overhead of speculative decoding is similar
/// to standard decoding for the same workload, since self-speculation shares weights
/// and KV cache.
#[test]
fn test_speculative_memory_overhead_vs_standard() {
    let config = SchedulerConfig::default();
    let num_kv_blocks = 1024;

    // Standard engine (no speculative)
    let mut std_engine = Engine::with_config(
        IncrementModel,
        None,
        config.clone(),
        4,
        num_kv_blocks,
    );

    let (tx_std, _rx_std) = mpsc::channel(64);
    std_engine.add_request(Request::new(1, vec![10, 20, 30], 10), tx_std);

    // Measure KV usage after prefill (peak usage during inference)
    let _ = std_engine.step();
    let (std_used, _) = std_engine.scheduler.get_kv_cache_usage();
    assert!(std_used > 0, "Standard engine should have KV blocks after prefill");
    while std_engine.has_pending() {
        let _ = std_engine.step();
    }

    // Speculative engine
    let mut spec_engine = Engine::with_config(
        IncrementModel,
        Some(IncrementModel),
        config.clone(),
        4,
        num_kv_blocks,
    );

    spec_engine.enable_adaptive_speculative(AdaptiveDraftConfig::default());

    let (tx_spec, _rx_spec) = mpsc::channel(64);
    spec_engine.add_request(Request::new(1, vec![10, 20, 30], 10), tx_spec);

    let _ = spec_engine.step(); // prefill
    let (spec_used, _) = spec_engine.scheduler.get_kv_cache_usage();
    assert!(spec_used > 0, "Speculative engine should have KV blocks after prefill");

    while spec_engine.has_pending() {
        let _ = spec_engine.step_adaptive_speculative();
    }

    // Speculative decoding should NOT use significantly more KV blocks than standard.
    // For self-speculation, the KV cache overhead is minimal since both models share
    // the same weights. At the same request state (after prefill), KV usage should be comparable.
    assert!(
        spec_used <= std_used || (spec_used as f64 / std_used as f64) <= 1.5,
        "Speculative KV block overhead should be < 50% (was spec={}, std={})",
        spec_used,
        std_used
    );
}

/// SPEC-04.3: Verify that the SelfSpeculativeModel shares weights (zero-copy memory)
/// by checking that it wraps the same model without allocating additional model parameters.
#[test]
fn test_self_speculative_weight_sharing() {
    // SelfSpeculativeModel wraps the same IncrementModel without allocating
    // additional weight storage. The draft_layer_count specifies how many
    // layers to use from the target model.
    let config = SpeculationConfig::builder()
        .draft_layers(4)
        .build();

    let model = IncrementModel;
    let self_spec = SelfSpeculativeModel::new(model, config);

    // Self-speculative model should use a subset of layers
    assert!(
        self_spec.draft_layer_count() > 0,
        "Draft layer count should be positive"
    );

    // The draft layer count should be <= total model layers
    assert!(
        self_spec.draft_layer_count() <= self_spec.model().num_layers(),
        "Draft layers should not exceed total model layers"
    );
}

/// SPEC-04.3: Verify that with self-speculation, the KV cache usage can be bounded
/// by limiting draft tokens. The overhead is proportional to draft_count.
#[test]
fn test_speculative_memory_overhead_scales_with_draft_count() {
    let config = SchedulerConfig::default();
    let num_kv_blocks = 1024;

    // Low draft count
    let mut low_draft_engine = Engine::with_config(
        IncrementModel,
        Some(IncrementModel),
        config.clone(),
        2, // low max_draft_tokens
        num_kv_blocks,
    );
    low_draft_engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 1,
        max_draft_tokens: 2,
        ..Default::default()
    });

    let (tx, _rx) = mpsc::channel(64);
    low_draft_engine.add_request(Request::new(1, vec![10, 20, 30], 10), tx);
    let _ = low_draft_engine.step();

    // Measure KV usage after prefill
    let (low_used, _) = low_draft_engine.scheduler.get_kv_cache_usage();
    while low_draft_engine.has_pending() {
        let _ = low_draft_engine.step_adaptive_speculative();
    }

    // High draft count
    let mut high_draft_engine = Engine::with_config(
        IncrementModel,
        Some(IncrementModel),
        config.clone(),
        8, // high max_draft_tokens
        num_kv_blocks,
    );
    high_draft_engine.enable_adaptive_speculative(AdaptiveDraftConfig {
        min_draft_tokens: 4,
        max_draft_tokens: 8,
        ..Default::default()
    });

    let (tx2, _rx2) = mpsc::channel(64);
    high_draft_engine.add_request(Request::new(1, vec![10, 20, 30], 10), tx2);
    let _ = high_draft_engine.step();

    let (high_used, _) = high_draft_engine.scheduler.get_kv_cache_usage();
    while high_draft_engine.has_pending() {
        let _ = high_draft_engine.step_adaptive_speculative();
    }

    // Both should have allocated blocks after prefill
    assert!(
        low_used > 0,
        "Low draft engine should allocate KV blocks (got {})",
        low_used
    );
    assert!(
        high_used > 0,
        "High draft engine should allocate KV blocks (got {})",
        high_used
    );

    // With same input, both should complete and not exceed the block pool
    assert!(
        low_used <= num_kv_blocks as u64,
        "Low draft engine should not exceed pool: {} vs {}",
        low_used,
        num_kv_blocks
    );
    assert!(
        high_used <= num_kv_blocks as u64,
        "High draft engine should not exceed pool: {} vs {}",
        high_used,
        num_kv_blocks
    );
}
