use serde::Serialize;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

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

pub struct MetricsCollector {
    tokens_total: Arc<AtomicU64>,
    requests_total: Arc<AtomicU64>,
    latencies: Arc<Mutex<Vec<f64>>>,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    start_time: std::time::Instant,
    requests_in_flight: Arc<AtomicU64>,
    kv_cache_blocks_used: Arc<AtomicU64>,
    kv_cache_blocks_total: Arc<AtomicU64>,
    prefix_cache_hits: Arc<AtomicU64>,
    prefix_cache_requests: Arc<AtomicU64>,
    prefill_tokens: Arc<AtomicU64>,
    decode_tokens: Arc<AtomicU64>,
    scheduler_wait_times: Arc<Mutex<Vec<f64>>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            tokens_total: Arc::new(AtomicU64::new(0)),
            requests_total: Arc::new(AtomicU64::new(0)),
            latencies: Arc::new(Mutex::new(Vec::new())),
            batch_sizes: Arc::new(Mutex::new(Vec::new())),
            start_time: std::time::Instant::now(),
            requests_in_flight: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_used: Arc::new(AtomicU64::new(0)),
            kv_cache_blocks_total: Arc::new(AtomicU64::new(0)),
            prefix_cache_hits: Arc::new(AtomicU64::new(0)),
            prefix_cache_requests: Arc::new(AtomicU64::new(0)),
            prefill_tokens: Arc::new(AtomicU64::new(0)),
            decode_tokens: Arc::new(AtomicU64::new(0)),
            scheduler_wait_times: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn record_tokens(&self, count: u64) {
        self.tokens_total.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_latency(&self, ms: f64) {
        if let Ok(mut latencies) = self.latencies.lock() {
            if latencies.capacity() == 0 {
                latencies.reserve(1000);
            }
            latencies.push(ms);
            if latencies.len() > 10000 {
                latencies.drain(0..1000);
            }
        }
    }

    pub fn record_batch_size(&self, size: usize) {
        if let Ok(mut sizes) = self.batch_sizes.lock() {
            if sizes.capacity() == 0 {
                sizes.reserve(1000);
            }
            sizes.push(size);
            if sizes.len() > 10000 {
                sizes.drain(0..1000);
            }
        }
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
        if let Ok(mut times) = self.scheduler_wait_times.lock() {
            if times.capacity() == 0 {
                times.reserve(1000);
            }
            times.push(ms);
            if times.len() > 10000 {
                times.drain(0..1000);
            }
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let tokens = self.tokens_total.load(Ordering::Relaxed);
        let requests = self.requests_total.load(Ordering::Relaxed);

        let (avg_latency, p50, p90, p99) = self
            .latencies
            .lock()
            .map(|latencies| {
                if latencies.is_empty() {
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    let len = latencies.len();
                    let sum: f64 = latencies.iter().sum();
                    let avg = sum / len as f64;

                    let mut sorted = Vec::with_capacity(len);
                    sorted.clone_from(&latencies);
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    fn get_p(xs: &[f64], p: f64) -> f64 {
                        xs[(p * xs.len() as f64).min(xs.len() as f64 - 1.0) as usize]
                    }

                    (
                        avg,
                        get_p(&sorted, 0.5),
                        get_p(&sorted, 0.9),
                        get_p(&sorted, 0.99),
                    )
                }
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0));

        let (avg_batch, current_batch) = self
            .batch_sizes
            .lock()
            .map(|sizes| {
                if sizes.is_empty() {
                    (0.0, 0)
                } else {
                    let sum: usize = sizes.iter().sum();
                    (sum as f64 / sizes.len() as f64, *sizes.last().unwrap_or(&0))
                }
            })
            .unwrap_or((0.0, 0));

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

        let avg_wait = self
            .scheduler_wait_times
            .lock()
            .map(|times| {
                if times.is_empty() {
                    0.0
                } else {
                    times.iter().sum::<f64>() / times.len() as f64
                }
            })
            .unwrap_or(0.0);

        MetricsSnapshot {
            tokens_total: tokens,
            requests_total: requests,
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

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

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
}
