// crates/core/src/metrics/collector/sampler/runtime.rs
//
// Lock-free delegation + CUDA Graph metrics + system counters.
// All methods here either forward to `self.runtime` (the lock-free hot path)
// or touch small `AtomicU64` fields / a single `DashMap` for inference latency.

use super::super::super::lock_free::MetricsSnapshot;
use super::EnhancedMetricsCollector;
use std::sync::atomic::Ordering;

impl EnhancedMetricsCollector {
    /// Snapshot for engine/API export (latency, throughput, KV usage).
    pub fn snapshot(&self) -> MetricsSnapshot {
        self.runtime.snapshot()
    }

    /// Accumulate generated token count into throughput counters.
    pub fn record_tokens(&self, count: u64) {
        self.runtime.record_tokens(count);
    }

    /// Split a scheduler `Batch` into prefill vs decode token counts and
    /// record both (RIL ISS-095).
    ///
    /// **Prefill tokens** = Σ input-token lengths of the sequences in the
    /// prefill phase (the vLLM convention: "prefill throughput" counts the
    /// prompt/chunk tokens *processed*). **Decode tokens** = the number of
    /// decode-phase sequences (each emits exactly one token per step).
    ///
    /// Called by every step path that composes a [`vllm_traits::Batch`]
    /// (regular, speculative, CUDA-graph) so the exported
    /// `prefill_throughput_tps` / `decode_throughput_tps` gauges are live
    /// instead of the pinned-0 they were stuck at when the only writers
    /// lived under `#[cfg(test)]`.
    pub fn record_batch_phase_tokens(&self, batch: &vllm_traits::Batch) {
        let mut prefill: u64 = 0;
        let mut decode: u64 = 0;
        for (i, is_prefill) in batch.is_prefill.iter().enumerate() {
            if *is_prefill {
                prefill +=
                    u64::try_from(batch.input_tokens.get(i).map_or(0, Vec::len)).unwrap_or(0);
            } else {
                decode += 1;
            }
        }
        if prefill > 0 {
            self.runtime.record_prefill_tokens(prefill);
        }
        if decode > 0 {
            self.runtime.record_decode_tokens(decode);
        }
    }

    /// Record the batch size of the most recent forward pass.
    pub fn record_batch_size(&self, size: usize) {
        self.runtime.record_batch_size(size);
    }

    /// Record end-to-end step latency in milliseconds.
    pub fn record_latency(&self, ms: f64) {
        self.runtime.record_latency(ms);
    }

    /// Record a scheduler queue→admission wait sample in milliseconds
    /// (RIL TASK-119). Called by the scheduler's batch builders for every
    /// sequence freshly drained from the request queue, so the exported
    /// `avg_scheduler_wait_time_ms` gauge reflects real admission delay.
    pub fn record_scheduler_wait_time(&self, ms: f64) {
        self.runtime.record_scheduler_wait_time(ms);
    }

    /// Update KV-cache occupancy counters (`used` blocks vs `total` capacity).
    pub fn record_kv_cache_usage(&self, used: u64, total: u64) {
        self.runtime.record_kv_cache_usage(used, total);
    }

    /// Snapshot the current number of complete prefix-cache entries.
    pub fn record_prefix_cache_nodes(&self, nodes: usize) {
        self.runtime.record_prefix_cache_nodes(nodes);
    }

    /// Increment prefix-cache hit counter after a radix-tree match.
    pub fn record_prefix_cache_hit(&self) {
        self.runtime.record_prefix_cache_hit();
    }

    /// Increment prefix-cache lookup counter (hit or miss).
    pub fn record_prefix_cache_request(&self) {
        self.runtime.record_prefix_cache_request();
    }

    /// Total prefix-cache hits since process start.
    pub fn prefix_cache_hits(&self) -> u64 {
        self.runtime.prefix_cache_hits()
    }

    /// Total prefix-cache lookups since process start.
    pub fn prefix_cache_requests(&self) -> u64 {
        self.runtime.prefix_cache_requests()
    }

    /// Increment CUDA Graph fast-path hit counter.
    pub fn record_cuda_graph_hit(&self) {
        self.cuda_graph_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment CUDA Graph miss counter (fell back to eager forward).
    pub fn record_cuda_graph_miss(&self) {
        self.cuda_graph_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the dropped-token counter (RIL ISS-074 / TASK-089).
    ///
    /// Called when the engine's `try_send` of a sampled token into the
    /// bounded response channel returns `TrySendError::Full` — the token is
    /// lost to the client stream. The counter makes the otherwise-silent
    /// loss visible on `/metrics` so operators can detect a consumer that
    /// systematically drains slower than generation.
    pub fn record_dropped_token(&self) {
        self.dropped_tokens_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the engine step-error counter (RIL ISS-090).
    ///
    /// Called from the run loop's single error site (where
    /// `Engine::error_count` is incremented), so `/debug/metrics`
    /// `errors_total` reflects real engine failures instead of the
    /// collector's fabricated `0` fallback.
    pub(crate) fn record_engine_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment completed-request counter.
    pub fn record_request(&self) {
        self.runtime.record_request();
    }

    /// Publish current waiting-queue depth for observability exporters.
    pub fn set_queue_depth(&self, depth: u64) {
        self.request_queue_depth.store(depth, Ordering::Relaxed);
    }

    /// Publish count of sequences currently in the running set.
    pub fn set_active_sequences(&self, count: u64) {
        self.active_sequences.store(count, Ordering::Relaxed);
    }

    /// Append a single inference latency sample (nanoseconds) to the rolling histogram.
    pub fn record_inference_latency(&self, duration_ns: u64) {
        let mut buckets = self
            .inference_latency_ns
            .entry("inference".to_string())
            .or_default();
        buckets.push(duration_ns);
        if buckets.len() > 10000 {
            buckets.remove(0);
        }
    }
}
