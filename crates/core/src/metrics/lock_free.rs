//! Lock-free counters and gauges used by the hot path of the metrics pipeline.
//!
//! Producers update atomic counters and push latency / batch-size /
//! scheduler-wait samples into bounded rolling windows (a `parking_lot`
//! `Mutex<VecDeque>` per window — an uncontended short-lived lock, never
//! a blocking send); the exporter reads a [`MetricsSnapshot`] via
//! `snapshot()`, which **clones** the windows without draining them, so
//! any number of consumers (the `/metrics` scrape, the engine's
//! `GetMetrics` round-trip, a future OTLP exporter) all observe the same
//! samples (RIL ISS-107).
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Snapshot of every observable engine metric at a single point in time. Fields cover throughput (tokens/sec), latency percentiles, scheduler queue depth, and KV-cache occupancy. Cloned and serialized on every metrics export.
#[derive(Debug, Clone, Serialize, Default)]
pub struct MetricsSnapshot {
    /// Cumulative tokens generated since process start.
    pub tokens_total: u64,
    /// Cumulative requests processed since process start.
    pub requests_total: u64,
    // RIL ISS-134: these measure ONE engine scheduler step (record_latency
    // fires once per step from batch.rs/dispatch.rs/graph_step.rs), NOT
    // end-to-end request latency — labeled honestly to stop operators
    // misreading step time as request time.
    /// Mean engine-step latency in milliseconds.
    pub avg_latency_ms: f64,
    /// 50th-percentile engine-step latency in milliseconds.
    pub p50_latency_ms: f64,
    /// 90th-percentile engine-step latency in milliseconds.
    pub p90_latency_ms: f64,
    /// 99th-percentile engine-step latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Mean batch size over the recent sampling window.
    pub avg_batch_size: f64,
    /// Batch size of the most recent scheduler step.
    pub current_batch_size: usize,
    /// Requests currently in the scheduler (waiting + running).
    pub requests_in_flight: u64,
    /// KV-cache blocks currently allocated.
    pub kv_cache_blocks_used: u64,
    /// Total KV-cache blocks available.
    pub kv_cache_blocks_total: u64,
    /// Fraction of KV-cache blocks currently in use.
    pub kv_cache_usage_percent: f64,
    /// Prefix-cache hit rate since process start.
    pub prefix_cache_hit_rate: f64,
    /// Number of complete entries currently in the prefix (radix) cache.
    pub prefix_cache_nodes: usize,
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
    /// Current number of complete prefix-cache (radix-tree) entries.
    prefix_cache_nodes: Arc<AtomicU64>,
    /// Cumulative prefill-phase tokens.
    prefill_tokens: Arc<AtomicU64>,
    /// Cumulative decode-phase tokens.
    decode_tokens: Arc<AtomicU64>,
    /// Process-start instant; basis for tokens/sec computation.
    start_time: std::time::Instant,

    /// Capacity of each bounded rolling window (latency / batch / wait).
    window_capacity: usize,
    /// Rolling latency samples (ms) since the process window began to fill.
    /// `snapshot()` clones; producers push under the window lock.
    latency_window: Mutex<VecDeque<f64>>,
    /// Rolling batch-size samples.
    batch_size_window: Mutex<VecDeque<usize>>,
    /// Rolling scheduler-wait samples (ms).
    scheduler_wait_window: Mutex<VecDeque<f64>>,
}

impl LockFreeMetrics {
    /// Construct a `LockFreeMetrics` whose bounded rolling windows hold up
    /// to `capacity` samples each for latency, batch size, and
    /// scheduler-wait time. When a window is full the oldest sample is
    /// evicted so the newest are retained (a "most recent N" window);
    /// producers never block and `snapshot()` never drains.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens_total: Arc::new(AtomicU64::new(0)),
            requests_total: Arc::new(AtomicU64::new(0)),
            requests_in_flight: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_used: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_total: Arc::new(AtomicU64::new(0)),
            prefix_cache_hits: Arc::new(AtomicU64::new(0)),
            prefix_cache_requests: Arc::new(AtomicU64::new(0)),
            prefix_cache_nodes: Arc::new(AtomicU64::new(0)),
            prefill_tokens: Arc::new(AtomicU64::new(0)),
            decode_tokens: Arc::new(AtomicU64::new(0)),
            start_time: std::time::Instant::now(),
            window_capacity: capacity,
            latency_window: Mutex::new(VecDeque::with_capacity(capacity)),
            batch_size_window: Mutex::new(VecDeque::with_capacity(capacity)),
            scheduler_wait_window: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    /// Push `value` onto `window`, evicting the oldest sample when the
    /// rolling window is at capacity. Bounded-memory, never blocks for a
    /// meaningful duration (an uncontended `parking_lot` lock).
    fn push_sample<T>(window: &Mutex<VecDeque<T>>, capacity: usize, value: T) {
        let mut samples = window.lock();
        if samples.len() >= capacity {
            samples.pop_front();
        }
        samples.push_back(value);
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

    /// Record a per-step latency sample in milliseconds. Pushed into the
    /// bounded rolling window; the oldest sample is evicted when the
    /// window is at capacity. Never blocks for a meaningful duration.
    pub fn record_latency(&self, ms: f64) {
        Self::push_sample(&self.latency_window, self.window_capacity, ms);
    }

    /// Record a per-step batch-size sample (bounded rolling window).
    pub fn record_batch_size(&self, size: usize) {
        Self::push_sample(&self.batch_size_window, self.window_capacity, size);
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

    /// Snapshot the current number of complete prefix-cache entries. Stored as
    /// an absolute count (not an increment); the engine refreshes it on each
    /// metrics request from the radix tree's live entry count.
    pub fn record_prefix_cache_nodes(&self, nodes: usize) {
        self.prefix_cache_nodes
            .store(nodes as u64, Ordering::Relaxed);
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

    /// Read atomics and the bounded rolling windows into a
    /// [`MetricsSnapshot`]. **Non-destructive** (RIL ISS-107): the
    /// latency / batch-size / wait-time windows are cloned, not drained,
    /// so any number of concurrent consumers observe the same samples —
    /// pre-fix `try_recv` drained the rings, so a second reader (a
    /// concurrent `/metrics` scrape, the engine `GetMetrics` round-trip,
    /// or a future OTLP exporter) silently stole the first reader's data.
    /// Lifetime atomic counters are never touched.
    #[must_use]
    // invariant: counters are bounded by uptime; u64/usize -> f64 precision
    // loss is acceptable for snapshot metrics (p50/p90/p99/throughput).
    #[allow(clippy::cast_precision_loss)]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let latencies: Vec<f64> = self.latency_window.lock().iter().copied().collect();
        let batch_sizes: Vec<usize> = self.batch_size_window.lock().iter().copied().collect();
        let wait_times: Vec<f64> = self.scheduler_wait_window.lock().iter().copied().collect();

        let (avg_latency, p50, p90, p99) = if latencies.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let sum: f64 = latencies.iter().sum();
            let avg = sum / latencies.len() as f64;

            // Sort in place — `latencies` is already an owned clone of the
            // window, so no second copy is needed for the percentiles.
            let mut sorted = latencies;
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

        let prefix_cache_nodes = self.prefix_cache_nodes.load(Ordering::Relaxed) as usize;

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
            kv_cache_blocks_used: kv_used,
            kv_cache_blocks_total: kv_total,
            kv_cache_usage_percent,
            prefix_cache_hit_rate,
            prefix_cache_nodes,
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

impl LockFreeMetrics {
    /// Mark the start of a request: increment the in-flight counter.
    /// Wired into [`crate::engine::Engine::add_request`] (admitted
    /// requests only) so `requests_in_flight` is a live gauge instead of
    /// the pinned 0 it was stuck at when the only writers lived under
    /// `#[cfg(test)]` (RIL ISS-082).
    pub(crate) fn record_request_start(&self) {
        self.requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark the end of a request: decrement the in-flight counter,
    /// **saturating at 0**. Pre-fix this was a raw `fetch_sub(1)`, so an
    /// unbalanced end (a sequence finalized twice, or cancelled without a
    /// recorded start) wrapped to `u64::MAX` (~1.8e19) and poisoned
    /// dashboards (RIL ISS-082). Paired with
    /// [`Self::record_request_start`] via `finalize_finished`.
    pub(crate) fn record_request_end(&self) {
        let _ = self
            .requests_in_flight
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    /// Add `count` to the lifetime prefill-tokens counter — the basis for
    /// the `prefill_throughput_tps` gauge (RIL ISS-095). Production callers
    /// go through `EnhancedMetricsCollector::record_batch_phase_tokens`,
    /// which reads a scheduler `Batch`'s phase flags and input-token
    /// lengths to split a step's work into prefill vs decode.
    pub(crate) fn record_prefill_tokens(&self, count: u64) {
        self.prefill_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Add `count` to the lifetime decode-tokens counter — the basis for
    /// the `decode_throughput_tps` gauge (RIL ISS-095).
    pub(crate) fn record_decode_tokens(&self, count: u64) {
        self.decode_tokens.fetch_add(count, Ordering::Relaxed);
    }

    /// Record a scheduler-wait-time sample in milliseconds (queue→admission
    /// delay). Pushed into the bounded rolling window (oldest evicted at
    /// capacity); never blocks for a meaningful duration.
    ///
    /// RIL TASK-119: production callers are the two scheduler batch builders
    /// (`build_batch` and `build_batch_with_graph`/`select_sequences_for_phase`),
    /// which now measure each drained sequence's `arrival_time`→now elapsed
    /// via `RequestQueue::drain_by_phase` and forward it here — previously the
    /// only recorder lived under `#[cfg(test)]`, so
    /// `avg_scheduler_wait_time_ms` was pinned at 0 in production (the
    /// ISS-095 documented-follow-up is now closed).
    pub(crate) fn record_scheduler_wait_time(&self, ms: f64) {
        Self::push_sample(&self.scheduler_wait_window, self.window_capacity, ms);
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
