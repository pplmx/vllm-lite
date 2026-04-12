use crossbeam::channel::{Receiver, Sender, bounded};
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Serialize, Default)]
pub struct MetricsSnapshot {
    pub tokens_total: u64,
    pub requests_total: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub avg_batch_size: f64,
    pub current_batch_size: usize,
    pub requests_in_flight: u64,
    pub kv_cache_usage_percent: f64,
    pub prefix_cache_hit_rate: f64,
    pub prefill_throughput: f64,
    pub decode_throughput: f64,
    pub avg_scheduler_wait_time_ms: f64,
}

pub struct LockFreeMetrics {
    tokens_total: Arc<AtomicU64>,
    requests_total: Arc<AtomicU64>,
    requests_in_flight: Arc<AtomicU64>,
    kv_cache_blocks_used: Arc<AtomicU64>,
    kv_cache_blocks_total: Arc<AtomicU64>,
    prefix_cache_hits: Arc<AtomicU64>,
    prefix_cache_requests: Arc<AtomicU64>,
    prefill_tokens: Arc<AtomicU64>,
    decode_tokens: Arc<AtomicU64>,
    start_time: std::time::Instant,

    latency_sender: Sender<f64>,
    latency_receiver: Receiver<f64>,
    batch_size_sender: Sender<usize>,
    batch_size_receiver: Receiver<usize>,
    scheduler_wait_sender: Sender<f64>,
    scheduler_wait_receiver: Receiver<f64>,
}

impl LockFreeMetrics {
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

    pub fn record_tokens(&self, count: u64) {
        self.tokens_total.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_latency(&self, ms: f64) {
        let _ = self.latency_sender.try_send(ms);
    }

    pub fn record_batch_size(&self, size: usize) {
        let _ = self.batch_size_sender.try_send(size);
    }

    pub fn record_request_start(&self) {
        self.requests_in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_end(&self) {
        self.requests_in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn record_kv_cache_usage(&self, used: u64, total: u64) {
        self.kv_cache_blocks_used.store(used, Ordering::Relaxed);
        self.kv_cache_blocks_total.store(total, Ordering::Relaxed);
    }

    pub fn record_prefix_cache_hit(&self) {
        self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefix_cache_request(&self) {
        self.prefix_cache_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefill_tokens(&self, count: u64) {
        self.prefill_tokens.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_decode_tokens(&self, count: u64) {
        self.decode_tokens.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_scheduler_wait_time(&self, ms: f64) {
        let _ = self.scheduler_wait_sender.try_send(ms);
    }

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
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
}

impl Default for LockFreeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub type MetricsCollector = LockFreeMetrics;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_snapshot_new_fields() {
        let collector = MetricsCollector::new();

        collector.record_request_start();
        collector.record_kv_cache_usage(50, 100);
        collector.record_prefix_cache_hit();
        collector.record_prefix_cache_request();
        collector.record_prefill_tokens(100);
        collector.record_decode_tokens(50);
        collector.record_scheduler_wait_time(10.0);
        collector.record_request_end();

        let snapshot = collector.snapshot();

        assert_eq!(snapshot.requests_in_flight, 0);
        assert!((snapshot.kv_cache_usage_percent - 50.0).abs() < 0.01);
        assert!((snapshot.prefix_cache_hit_rate - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_kv_cache_zero_total() {
        let collector = MetricsCollector::new();
        collector.record_kv_cache_usage(10, 0);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.kv_cache_usage_percent, 0.0);
    }

    #[test]
    fn test_metrics_prefix_cache_no_requests() {
        let collector = MetricsCollector::new();

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.prefix_cache_hit_rate, 0.0);
    }

    #[test]
    fn test_lock_free_metrics_single_record() {
        let collector = LockFreeMetrics::with_capacity(1024);
        collector.record_latency(10.5);

        let snapshot = collector.snapshot();
        assert!((snapshot.avg_latency_ms - 10.5).abs() < 0.01);
    }

    #[test]
    fn test_lock_free_metrics_burst_records() {
        let collector = LockFreeMetrics::with_capacity(1024);

        for i in 1..=100 {
            collector.record_latency(i as f64);
        }

        let snapshot = collector.snapshot();
        assert!((snapshot.avg_latency_ms - 50.5).abs() < 0.01);
        assert!((snapshot.p50_latency_ms - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_lock_free_metrics_buffer_overflow() {
        let collector = LockFreeMetrics::with_capacity(10);

        for i in 0..100 {
            collector.record_latency(i as f64);
        }

        let snapshot = collector.snapshot();
        assert!(snapshot.avg_latency_ms > 0.0);
    }
}
