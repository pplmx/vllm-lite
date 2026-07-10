//! Metrics sampling pipeline: receive raw events from workers, bucketise them, and publish aggregated snapshots to the `MetricsExporter`.
//!
//! The collector runs on a dedicated thread, draining the channel with
//! `try_recv` to keep latency low and falling through to a periodic flush
//! when the channel is idle. Exposed via [`crate::metrics::EnhancedMetricsCollector`].

// crates/core/src/metrics/collector/sampler/mod.rs
//
// Unified metrics collector: implements the runtime sampling and recording
// for the engine, scheduler, and HTTP exporters. Method bodies are split
// across the following submodules by concern:
// - `runtime`     — lock-free delegation + CUDA Graph + system counters
// - `packing`     — packing efficiency / waste metrics
// - `speculative` — speculative decoding + per-request acceptance
// - `draft`       — v18.0 multi-model spec draft-resolution metrics

use super::super::lock_free::LockFreeMetrics;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use vllm_traits::SeqId;

// Re-exported so `mod tests` (via `use super::*;`) can reference the type
// without importing it directly. `draft.rs` already imports it locally.
#[cfg(test)]
pub use super::metrics::DraftResolutionKind;

mod draft;
mod packing;
mod runtime;
mod speculative;

#[derive(Debug)]
/// Unified metrics collector for scheduler, engine, and HTTP export.
pub struct EnhancedMetricsCollector {
    /// Lock-free counters for the hottest path (request counts, latencies, etc.).
    runtime: LockFreeMetrics,
    /// Total CUDA-Graph replay hits since process start.
    cuda_graph_hits: AtomicU64,
    /// Total CUDA-Graph cache misses (re-captures) since process start.
    cuda_graph_misses: AtomicU64,
    /// Total sequences packed into non-trivial batches.
    packing_sequences: AtomicU64,
    /// Total adaptive-speculation adjustments applied.
    speculative_adjustments: AtomicU64,
    /// Total error events observed across the engine + scheduler.
    errors_total: AtomicU64,
    /// Packing waste ratio (bits stored / bits useful) × 1000 (fixed-point).
    packing_waste_ratio: AtomicU64,
    /// Packing efficiency percentage × 100 (fixed-point).
    packing_efficiency: AtomicU64,
    /// Speculative acceptance rate percentage × 100 (fixed-point).
    speculative_acceptance_rate: AtomicU64,
    /// Total draft tokens emitted since process start.
    speculative_draft_count: AtomicU64,
    /// Speculative efficiency (accepted / drafted) × 1000 (fixed-point).
    speculative_efficiency: AtomicU64,
    /// Throughput speedup vs. greedy decoding × 100 (fixed-point).
    throughput_speedup_ratio: AtomicU64,
    /// Current depth of the scheduler's request queue.
    request_queue_depth: AtomicU64,
    /// Current number of active (running) sequences.
    active_sequences: AtomicU64,
    /// Cumulative speculative-token count across all requests.
    speculative_per_request_count: AtomicU64,
    // v18.0 multi-model speculative decoding
    /// Drafts resolved via the external registry.
    draft_resolutions_external_total: AtomicU64,
    /// Drafts resolved via self-speculation.
    draft_resolutions_self_spec_total: AtomicU64,
    /// Requests that fell back to no-draft.
    draft_resolutions_none_total: AtomicU64,
    /// Total draft-loader failures (model missing / OOM / corrupt).
    draft_load_failures_total: AtomicU64,
    /// Total runtime errors thrown by the draft backend.
    draft_runtime_errors_total: AtomicU64,
    /// Per-endpoint inference-latency histograms (nanoseconds).
    inference_latency_ns: DashMap<String, Vec<u64>>,
    /// Per-request accepted / drafted token tallies.
    per_request_acceptance: DashMap<SeqId, (AtomicU64, AtomicU64)>,
}

impl EnhancedMetricsCollector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            runtime: LockFreeMetrics::new(),
            cuda_graph_hits: AtomicU64::new(0),
            cuda_graph_misses: AtomicU64::new(0),
            packing_sequences: AtomicU64::new(0),
            speculative_adjustments: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            packing_waste_ratio: AtomicU64::new(0),
            packing_efficiency: AtomicU64::new(0),
            speculative_acceptance_rate: AtomicU64::new(0),
            speculative_draft_count: AtomicU64::new(0),
            speculative_efficiency: AtomicU64::new(0),
            throughput_speedup_ratio: AtomicU64::new(0),
            request_queue_depth: AtomicU64::new(0),
            active_sequences: AtomicU64::new(0),
            speculative_per_request_count: AtomicU64::new(0),
            draft_resolutions_external_total: AtomicU64::new(0),
            draft_resolutions_self_spec_total: AtomicU64::new(0),
            draft_resolutions_none_total: AtomicU64::new(0),
            draft_load_failures_total: AtomicU64::new(0),
            draft_runtime_errors_total: AtomicU64::new(0),
            inference_latency_ns: DashMap::new(),
            per_request_acceptance: DashMap::new(),
        }
    }

    // Getters for testing and export
    pub fn get_counter(&self, name: &str) -> u64 {
        match name {
            "cuda_graph_hits_total" => self.cuda_graph_hits.load(Ordering::Relaxed),
            "cuda_graph_misses_total" => self.cuda_graph_misses.load(Ordering::Relaxed),
            "packing_sequences_total" => self.packing_sequences.load(Ordering::Relaxed),
            "speculative_adjustments_total" => self.speculative_adjustments.load(Ordering::Relaxed),
            "requests_total" => self.runtime.requests_total(),
            "errors_total" => self.errors_total.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    pub fn get_gauge(&self, name: &str) -> u64 {
        match name {
            "packing_efficiency" => self.packing_efficiency.load(Ordering::Relaxed),
            "packing_waste_ratio" => self.packing_waste_ratio.load(Ordering::Relaxed),
            "speculative_acceptance_rate" => {
                self.speculative_acceptance_rate.load(Ordering::Relaxed)
            }
            "speculative_draft_count" => self.speculative_draft_count.load(Ordering::Relaxed),
            "speculative_efficiency" => self.speculative_efficiency.load(Ordering::Relaxed),
            "throughput_speedup_ratio" => self.throughput_speedup_ratio.load(Ordering::Relaxed),
            "request_queue_depth" => self.request_queue_depth.load(Ordering::Relaxed),
            "active_sequences" => self.active_sequences.load(Ordering::Relaxed),
            "speculative_per_request_count" => {
                self.speculative_per_request_count.load(Ordering::Relaxed)
            }
            _ => 0,
        }
    }
}

impl Default for EnhancedMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (cuda_graph_hit counter, packing_efficiency / speculative_acceptance /
// speculative_efficiency / throughput_speedup gauges, inference
// latency samples, draft-resolution metrics counters and
// DraftResolutionKind parse / Display).
#[cfg(test)]
mod tests;
