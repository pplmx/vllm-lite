use serde::Serialize;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Serialize)]
pub struct MetricsSnapshot {
    pub tokens_total: u64,
    pub requests_total: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub avg_batch_size: f64,
    pub current_batch_size: usize,
}

pub struct MetricsCollector {
    tokens_total: Arc<AtomicU64>,
    requests_total: Arc<AtomicU64>,
    latencies: Arc<Mutex<Vec<f64>>>,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    #[allow(dead_code)]
    start_time: std::time::Instant,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            tokens_total: Arc::new(AtomicU64::new(0)),
            requests_total: Arc::new(AtomicU64::new(0)),
            latencies: Arc::new(Mutex::new(Vec::new())),
            batch_sizes: Arc::new(Mutex::new(Vec::new())),
            start_time: std::time::Instant::now(),
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
            latencies.push(ms);
            if latencies.len() > 10000 {
                latencies.drain(0..1000);
            }
        }
    }

    pub fn record_batch_size(&self, size: usize) {
        if let Ok(mut sizes) = self.batch_sizes.lock() {
            sizes.push(size);
            if sizes.len() > 10000 {
                sizes.drain(0..1000);
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

        MetricsSnapshot {
            tokens_total: tokens,
            requests_total: requests,
            avg_latency_ms: avg_latency,
            p50_latency_ms: p50,
            p90_latency_ms: p90,
            p99_latency_ms: p99,
            avg_batch_size: avg_batch,
            current_batch_size: current_batch,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
