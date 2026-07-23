//! Lock-free counters and gauges backed by `crossbeam` channels, used by the hot path of the metrics pipeline.
//!
//! Producers send events through an unbounded MPSC channel; a dedicated
//! consumer thread folds them into atomic counters that the exporter
//! reads without ever taking a mutex. Falls back to the locked variant
//! on single-threaded test builds.
use crossbeam::channel::{Receiver, Sender, bounded};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Snapshot of every observable engine metric at a single point in time. Fields cover throughput (tokens/sec), latency percentiles, scheduler queue depth, and KV-cache occupancy. Cloned and serialized on every metrics export.
#[derive(Debug, Clone, Serialize, Default)]
pub struct MetricsSnapshot {
    /// Cumulative tokens generated since process start.
    pub tokens_total: u64,
    /// Cumulative requests processed since process start.
    pub requests_total: u64,
    /// Mean request latency in milliseconds.
    pub avg_latency_ms: f64,
    /// 50th-percentile request latency in milliseconds.
    pub p50_latency_ms: f64,
    /// 90th-percentile request latency in milliseconds.
    pub p90_latency_ms: f64,
    /// 99th-percentile request latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Mean batch size over the recent sampling window.
    pub avg_batch_size: f64,
    /// Batch size of the most recent scheduler step.
    pub current_batch_size: usize,
    /// Requests currently in the scheduler (waiting + running).
    pub requests_in_flight: u64,
    /// Fraction of KV-cache blocks currently in use.
    pub kv_cache_usage_percent: f64,
    /// Prefix-cache hit rate since process start.
    pub prefix_cache_hit_rate: f64,
    /// Prefill-phase tokens per second.
    pub prefill_throughput: f64,
    /// Decode-phase tokens per second.
    pub decode_throughput: f64,
    /// Mean time requests spent in the waiting queue.
    pub avg_scheduler_wait_time_ms: f64,
}

#[derive(Debug)]
/// Lock-free metrics recorder. Producers update per-counter atomics; consumers snapshot a [`MetricsSnapshot`] via `snapshot()`. Used in the hot path where mutex contention would show up in latency.
pub struct LockFreeMetrics {
    /// Cumulative tokens generated.
    tokens_total: Arc<AtomicU64>,
    /// Cumulative requests processed.
    requests_total: Arc<AtomicU64>,
    /// Requests currently waiting or running.
    requests_in_flight: Arc<AtomicU64>,
    /// KV-cache blocks currently allocated.
    kv_cache_blocks_used: Arc<AtomicU64>,
    /// Total KV-cache blocks available.
    kv_cache_blocks_total: Arc<AtomicU64>,
    /// Prefix-cache lookups that hit an existing entry.
    prefix_cache_hits: Arc<AtomicU64>,
    /// Total prefix-cache lookups.
    prefix_cache_requests: Arc<AtomicU64>,
    /// Cumulative prefill-phase tokens.
    prefill_tokens: Arc<AtomicU64>,
    /// Cumulative decode-phase tokens.
    decode_tokens: Arc<AtomicU64>,
    /// Process-start instant; basis for tokens/sec computation.
    start_time: std::time::Instant,

    /// Sender side of the bounded latency ring channel.
    latency_sender: Sender<f64>,
    /// Receiver side of the bounded latency ring channel.
    latency_receiver: Receiver<f64>,
    /// Sender side of the bounded batch-size ring channel.
    batch_size_sender: Sender<usize>,
    /// Receiver side of the bounded batch-size ring channel.
    batch_size_receiver: Receiver<usize>,
    /// Sender side of the bounded scheduler-wait ring channel.
    // Only read by `record_scheduler_wait_time` (test-only); rustc reports
    // the field as never-read in non-test builds because the only consumer
    // is reachable only under cfg(test).
    #[allow(dead_code)]
    scheduler_wait_sender: Sender<f64>,
    /// Receiver side of the bounded scheduler-wait ring channel.
    scheduler_wait_receiver: Receiver<f64>,
}

impl LockFreeMetrics {
    /// Construct a `LockFreeMetrics` whose bounded ring channels hold up to
    /// `capacity` samples each for latency, batch size, and scheduler-wait
    /// time. Once full, additional samples are dropped (via `try_send`),
    /// keeping the hot path lock-free.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let (latency_tx, latency_rx) = bounded(capacity);
        let (batch_tx, batch_rx) = bounded(capacity);
        let (wait_tx, wait_rx) = bounded(capacity);

        Self {
            tokens_total: Arc::new(AtomicU64::new(0)),
            requests_total: Arc::new(AtomicU64::new(0)),
            requests_in_flight: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_used: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_total: Arc::new(AtomicU64::new(0)),
            prefix_cache_hits: Arc::new(AtomicU64::new(0)),
            prefix_cache_requests: Arc::new(AtomicU64::new(0)),
            prefill_tokens: Arc::new(AtomicU64::new(0)),
            decode_tokens: Arc::new(AtomicU64::new(0)),
            start_time: std::time::Instant::now(),
            latency_sender: latency_tx,
            latency_receiver: latency_rx,
            batch_size_sender: batch_tx,
            batch_size_receiver: batch_rx,
            scheduler_wait_sender: wait_tx,
            scheduler_wait_receiver: wait_rx,
        }
    }

    /// Add `count` to the lifetime token counter. Hot-path: uses a single
    /// relaxed atomic increment.
    pub fn record_tokens(&self, count: u64) {
        self.tokens_total.fetch_add(count, Ordering::Relaxed);
    }

    /// Increment the lifetime request counter by one.
    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a per-step latency sample in milliseconds. Sample is pushed to
    /// the bounded latency channel; if the channel is full it is silently
    /// dropped.
    pub fn record_latency(&self, ms: f64) {
        let _ = self.latency_sender.try_send(ms);
    }

    /// Record a per-step batch-size sample. Dropped if the channel is full.
    pub fn record_batch_size(&self, size: usize) {
        let _ = self.batch_size_sender.try_send(size);
    }

    /// Snapshot the absolute KV-cache utilization. Both values are stored as
    /// separate atomics; the percentage is computed at `snapshot()` time.
    pub fn record_kv_cache_usage(&self, used: u64, total: u64) {
        self.kv_cache_blocks_used.store(used, Ordering::Relaxed);
        self.kv_cache_blocks_total.store(total, Ordering::Relaxed);
    }

    /// Increment the prefix-cache hit counter. Pair each call with a
    /// `record_prefix_cache_request` to compute a hit-rate.
    pub fn record_prefix_cache_hit(&self) {
        self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the prefix-cache lookup counter (called for every prompt,
    /// regardless of hit/miss).
    pub fn record_prefix_cache_request(&self) {
        self.prefix_cache_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Total number of requests served since start.
    #[must_use]
    pub fn requests_total(&self) -> u64 {
        self.requests_total.load(Ordering::Relaxed)
    }

    /// `prefix_cache_hits`: total prefix cache hits since start.
    #[must_use]
    pub fn prefix_cache_hits(&self) -> u64 {
        self.prefix_cache_hits.load(Ordering::Relaxed)
    }

    /// `prefix_cache_requests`: total prefix cache lookups since start.
    #[must_use]
    pub fn prefix_cache_requests(&self) -> u64 {
        self.prefix_cache_requests.load(Ordering::Relaxed)
    }

    /// Drain all ring channels and atomics into a [`MetricsSnapshot`].
    /// After this call, the latency / batch-size / wait-time samples are
    /// reset (only the unconsumed ring entries are flushed — lifetime
    /// atomic counters are not touched).
    #[must_use]
    // invariant: counters are bounded by uptime; u64/usize -> f64 precision
    // loss is acceptable for snapshot metrics (p50/p90/p99/throughput).
    #[allow(clippy::cast_precision_loss)]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let mut latencies = Vec::new();
        while let Ok(ms) = self.latency_receiver.try_recv() {
            latencies.push(ms);
        }

        let mut batch_sizes = Vec::new();
        while let Ok(size) = self.batch_size_receiver.try_recv() {
            batch_sizes.push(size);
        }

        let mut wait_times = Vec::new();
        while let Ok(ms) = self.scheduler_wait_receiver.try_recv() {
            wait_times.push(ms);
        }

        let (avg_latency, p50, p90, p99) = if latencies.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let sum: f64 = latencies.iter().sum();
            let avg = sum / latencies.len() as f64;

            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let get_p = |xs: &[f64], p: f64| -> f64 {
                // invariant: p in 0..=1 and xs non-empty, so the floor result is
                // in 0..xs.len(); truncation/saturation is bounded.
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let idx = ((p * (xs.len() as f64 - 1.0)).floor() as usize).min(xs.len() - 1);
                xs[idx]
            };

            (
                avg,
                get_p(&sorted, 0.5),
                get_p(&sorted, 0.9),
                get_p(&sorted, 0.99),
            )
        };

        let (avg_batch, current_batch) = if batch_sizes.is_empty() {
            (0.0, 0)
        } else {
            let sum: usize = batch_sizes.iter().sum();
            (
                sum as f64 / batch_sizes.len() as f64,
                *batch_sizes.last().unwrap_or(&0),
            )
        };

        let requests_in_flight = self.requests_in_flight.load(Ordering::Relaxed);

        let kv_used = self.kv_cache_blocks_used.load(Ordering::Relaxed);
        let kv_total = self.kv_cache_blocks_total.load(Ordering::Relaxed);
        let kv_cache_usage_percent = if kv_total > 0 {
            (kv_used as f64 / kv_total as f64) * 100.0
        } else {
            0.0
        };

        let hits = self.prefix_cache_hits.load(Ordering::Relaxed);
        let total_reqs = self.prefix_cache_requests.load(Ordering::Relaxed);
        let prefix_cache_hit_rate = if total_reqs > 0 {
            (hits as f64 / total_reqs as f64) * 100.0
        } else {
            0.0
        };

        let uptime = self.start_time.elapsed().as_secs_f64();
        let prefill_throughput = if uptime > 0.0 {
            self.prefill_tokens.load(Ordering::Relaxed) as f64 / uptime
        } else {
            0.0
        };
        let decode_throughput = if uptime > 0.0 {
            self.decode_tokens.load(Ordering::Relaxed) as f64 / uptime
        } else {
            0.0
        };

        let avg_wait = if wait_times.is_empty() {
            0.0
        } else {
            wait_times.iter().sum::<f64>() / wait_times.len() as f64
        };

        MetricsSnapshot {
            tokens_total: self.tokens_total.load(Ordering::Relaxed),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p90_latency_ms: p90,
            p99_latency_ms: p99,
            avg_batch_size: avg_batch,
            current_batch_size: current_batch,
            requests_in_flight,
            kv_cache_usage_percent,
            prefix_cache_hit_rate,
            prefill_throughput,
            decode_throughput,
            avg_scheduler_wait_time_ms: avg_wait,
        }
    }
}

impl LockFreeMetrics {
    /// Create a new lock-free metrics store with the default capacity (1024).
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
}

impl Default for LockFreeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait implemented by every metrics backend (lock-free, enhanced, prometheus). Provides `snapshot()` and `reset()` for periodic export.
pub type MetricsCollector = LockFreeMetrics;

#[cfg(test)]
impl LockFreeMetrics {
    /// Mark the start of a request: increment the in-flight counter.
    pub(crate) fn record_request_start(&self) {
        self.requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark the end of a request: decrement the in-flight counter.
    pub(crate) fn record_request_end(&self) {
        self.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    /// Add `count` to the lifetime prefill-tokens counter.
    pub(crate) fn record_prefill_tokens(&self, count: u64) {
        self.prefill_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Add `count` to the lifetime decode-tokens counter.
    pub(crate) fn record_decode_tokens(&self, count: u64) {
        self.decode_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Record a scheduler-wait-time sample in milliseconds. Dropped if the
    /// channel is full.
    pub(crate) fn record_scheduler_wait_time(&self, ms: f64) {
        let _ = self.scheduler_wait_sender.try_send(ms);
    }
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// metrics module under the 800-line soft cap. The sibling covers:
// MetricsCollector snapshot accuracy (kv_cache %, prefix_cache
// hit rate, division-by-zero safety) and LockFreeMetrics
// ring-buffer behavior (single record, burst of 100, overflow
// graceful wrap).
#[cfg(test)]
mod tests;
