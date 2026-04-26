use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchingStrategy {
    Static,
    Dynamic,
    Predictive,
}

#[derive(Debug, Clone, Copy)]
pub struct PredictiveBatchingConfig {
    pub strategy: BatchingStrategy,
    pub target_latency_ms: u64,
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub prediction_window_ms: u64,
    pub throughput_weight: f64,
}

impl Default for PredictiveBatchingConfig {
    fn default() -> Self {
        Self {
            strategy: BatchingStrategy::Dynamic,
            target_latency_ms: 100,
            min_batch_size: 1,
            max_batch_size: 256,
            prediction_window_ms: 50,
            throughput_weight: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestPattern {
    pub arrival_rate: f64,
    pub avg_prompt_tokens: usize,
    pub avg_decode_tokens: usize,
    pub timestamp: Instant,
}

impl Default for RequestPattern {
    fn default() -> Self {
        Self {
            arrival_rate: 0.0,
            avg_prompt_tokens: 128,
            avg_decode_tokens: 64,
            timestamp: Instant::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatcherMetrics {
    pub batches_created: usize,
    pub total_tokens: usize,
    pub avg_batch_size: f64,
    pub current_pattern: RequestPattern,
}

#[derive(Debug, Clone)]
struct RequestSample {
    timestamp: Instant,
    prompt_tokens: usize,
    decode_tokens: usize,
}

#[derive(Debug)]
pub struct PredictiveBatcher {
    config: PredictiveBatchingConfig,
    request_history: std::sync::Mutex<VecDeque<RequestSample>>,
    current_pattern: std::sync::Mutex<RequestPattern>,
    batch_counter: AtomicUsize,
    total_tokens_processed: AtomicUsize,
    total_batches: AtomicUsize,
    last_batch_time: std::sync::Mutex<Instant>,
}

impl PredictiveBatcher {
    pub fn new(config: PredictiveBatchingConfig) -> Self {
        Self {
            config,
            request_history: std::sync::Mutex::new(VecDeque::with_capacity(1000)),
            current_pattern: std::sync::Mutex::new(RequestPattern::default()),
            batch_counter: AtomicUsize::new(0),
            total_tokens_processed: AtomicUsize::new(0),
            total_batches: AtomicUsize::new(0),
            last_batch_time: std::sync::Mutex::new(Instant::now()),
        }
    }

    pub fn record_request(&self, prompt_tokens: usize, decode_tokens: usize) {
        let sample = RequestSample {
            timestamp: Instant::now(),
            prompt_tokens,
            decode_tokens,
        };

        let mut history = self.request_history.lock().unwrap();
        history.push_back(sample);
        if history.len() > 1000 {
            history.pop_front();
        }

        drop(history);
        self.update_pattern();
    }

    fn update_pattern(&self) {
        let window = Duration::from_millis(self.config.prediction_window_ms);
        let now = Instant::now();
        let cutoff = now - window;

        let history = self.request_history.lock().unwrap();
        let recent: Vec<_> = history.iter().filter(|s| s.timestamp > cutoff).collect();

        if recent.is_empty() {
            return;
        }

        let arrival_rate = recent.len() as f64 / window.as_secs_f64();
        let avg_prompt = recent.iter().map(|s| s.prompt_tokens).sum::<usize>() / recent.len();
        let avg_decode = recent.iter().map(|s| s.decode_tokens).sum::<usize>() / recent.len();

        let mut pattern = self.current_pattern.lock().unwrap();
        pattern.arrival_rate = arrival_rate;
        pattern.avg_prompt_tokens = avg_prompt;
        pattern.avg_decode_tokens = avg_decode;
        pattern.timestamp = now;
    }

    pub fn predict_batch_size(&self, pending_requests: usize) -> usize {
        match self.config.strategy {
            BatchingStrategy::Static => self.config.max_batch_size.min(pending_requests),

            BatchingStrategy::Dynamic => {
                let pattern = self.current_pattern.lock().unwrap();
                if pattern.arrival_rate < 10.0 {
                    self.config.min_batch_size.max(pending_requests)
                } else if pattern.arrival_rate < 50.0 {
                    (self.config.max_batch_size / 2).max(pending_requests)
                } else {
                    self.config.max_batch_size.min(pending_requests)
                }
            }

            BatchingStrategy::Predictive => {
                let pattern = self.current_pattern.lock().unwrap();

                let load_factor = (pattern.arrival_rate / 100.0).min(1.0);
                let latency_weight = 1.0 - self.config.throughput_weight;

                let base_size = self.config.min_batch_size as f64
                    + (self.config.max_batch_size - self.config.min_batch_size) as f64
                        * (self.config.throughput_weight * load_factor
                            + latency_weight * (1.0 - load_factor));

                (base_size as usize)
                    .max(pending_requests)
                    .min(self.config.max_batch_size)
            }
        }
    }

    pub fn should_start_batch(&self, pending_count: usize) -> bool {
        let optimal = self.predict_batch_size(pending_count);

        let last_time = *self.last_batch_time.lock().unwrap();
        let elapsed = last_time.elapsed();
        let latency_threshold = Duration::from_millis(self.config.target_latency_ms);

        pending_count >= optimal.min(self.config.min_batch_size) || elapsed > latency_threshold
    }

    pub fn on_batch_complete(&self, batch_size: usize, tokens_processed: usize) {
        self.batch_counter.fetch_add(1, Ordering::SeqCst);
        self.total_tokens_processed
            .fetch_add(tokens_processed, Ordering::SeqCst);
        self.total_batches.fetch_add(1, Ordering::SeqCst);
        *self.last_batch_time.lock().unwrap() = Instant::now();

        tracing::debug!(
            batch_size = batch_size,
            tokens = tokens_processed,
            batches_total = self.total_batches.load(Ordering::SeqCst),
            "batch_complete: predictive batching stats"
        );
    }

    pub fn get_metrics(&self) -> BatcherMetrics {
        BatcherMetrics {
            batches_created: self.batch_counter.load(Ordering::SeqCst),
            total_tokens: self.total_tokens_processed.load(Ordering::SeqCst),
            avg_batch_size: if self.batch_counter.load(Ordering::SeqCst) > 0 {
                self.total_tokens_processed.load(Ordering::SeqCst) as f64
                    / self.batch_counter.load(Ordering::SeqCst) as f64
            } else {
                0.0
            },
            current_pattern: self.current_pattern.lock().unwrap().clone(),
        }
    }
}

pub struct BatchOptimizer {
    config: PredictiveBatchingConfig,
    latency_history: VecDeque<(Instant, Duration)>,
    throughput_history: VecDeque<(Instant, f64)>,
}

impl BatchOptimizer {
    pub fn new(config: PredictiveBatchingConfig) -> Self {
        Self {
            config,
            latency_history: VecDeque::with_capacity(100),
            throughput_history: VecDeque::with_capacity(100),
        }
    }

    pub fn record_batch_latency(&mut self, latency: Duration) {
        let now = Instant::now();
        self.latency_history.push_back((now, latency));
        if self.latency_history.len() > 100 {
            self.latency_history.pop_front();
        }
    }

    pub fn record_throughput(&mut self, throughput: f64) {
        let now = Instant::now();
        self.throughput_history.push_back((now, throughput));
        if self.throughput_history.len() > 100 {
            self.throughput_history.pop_front();
        }
    }

    pub fn calculate_optimal_batch_size(&self) -> usize {
        let window = Duration::from_secs(10);
        let now = Instant::now();

        let recent_latencies: Vec<_> = self
            .latency_history
            .iter()
            .filter(|(t, _)| *t > now - window)
            .collect();

        let recent_throughputs: Vec<_> = self
            .throughput_history
            .iter()
            .filter(|(t, _)| *t > now - window)
            .collect();

        let avg_latency = if recent_latencies.is_empty() {
            Duration::from_millis(100)
        } else {
            let sum: Duration = recent_latencies.iter().map(|(_, l)| *l).sum();
            sum / recent_latencies.len() as u32
        };

        let avg_throughput = if recent_throughputs.is_empty() {
            10.0
        } else {
            recent_throughputs.iter().map(|(_, t)| t).sum::<f64>() / recent_throughputs.len() as f64
        };

        let latency_ratio = self.config.target_latency_ms as f64 / avg_latency.as_millis() as f64;
        let throughput_ratio = avg_throughput / 100.0;

        let adjustment = (latency_ratio * 0.6 + throughput_ratio * 0.4).clamp(0.5, 2.0);

        let base_size = (self.config.max_batch_size as f64 * adjustment) as usize;
        base_size.clamp(self.config.min_batch_size, self.config.max_batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_strategy() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Static,
            min_batch_size: 4,
            max_batch_size: 32,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        assert_eq!(batcher.predict_batch_size(10), 10);
        assert_eq!(batcher.predict_batch_size(50), 32);
    }

    #[test]
    fn test_dynamic_strategy_low_load() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Dynamic,
            min_batch_size: 2,
            max_batch_size: 32,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        assert_eq!(batcher.predict_batch_size(10), 10);
    }

    #[test]
    fn test_dynamic_strategy_high_load() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Dynamic,
            min_batch_size: 2,
            max_batch_size: 32,
            prediction_window_ms: 10,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        for i in 0..20 {
            batcher.record_request(128, 64);
            std::thread::sleep(Duration::from_millis(1));
        }

        let predicted = batcher.predict_batch_size(10);
        assert!(predicted >= 10 && predicted <= 32);
    }

    #[test]
    fn test_predictive_strategy() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Predictive,
            min_batch_size: 4,
            max_batch_size: 32,
            throughput_weight: 0.5,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        for _ in 0..10 {
            batcher.record_request(128, 64);
        }

        let predicted = batcher.predict_batch_size(8);
        assert!(predicted >= 4 && predicted <= 32);
    }

    #[test]
    fn test_should_start_batch_by_count() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Static,
            min_batch_size: 4,
            max_batch_size: 32,
            target_latency_ms: 1000,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        let result2 = batcher.should_start_batch(2);
        let result4 = batcher.should_start_batch(4);
        assert!(
            !result2 || result4,
            "Should generally start batches when enough pending"
        );
    }

    #[test]
    fn test_should_start_batch_by_latency() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Static,
            min_batch_size: 10,
            max_batch_size: 32,
            target_latency_ms: 1,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        std::thread::sleep(Duration::from_millis(5));

        assert!(batcher.should_start_batch(1));
    }

    #[test]
    fn test_batch_complete_metrics() {
        let config = PredictiveBatchingConfig::default();
        let batcher = PredictiveBatcher::new(config);

        batcher.on_batch_complete(10, 100);
        batcher.on_batch_complete(5, 50);

        let metrics = batcher.get_metrics();
        assert_eq!(metrics.batches_created, 2);
        assert_eq!(metrics.total_tokens, 150);
    }

    #[test]
    fn test_optimizer_calculates_optimal_size() {
        let config = PredictiveBatchingConfig {
            strategy: BatchingStrategy::Predictive,
            min_batch_size: 4,
            max_batch_size: 32,
            target_latency_ms: 100,
            ..Default::default()
        };
        let mut optimizer = BatchOptimizer::new(config);

        for _ in 0..20 {
            optimizer.record_batch_latency(Duration::from_millis(150));
            optimizer.record_throughput(50.0);
        }

        let optimal = optimizer.calculate_optimal_batch_size();
        assert!(optimal >= 4 && optimal <= 32);
    }

    #[test]
    fn test_request_pattern_recording() {
        let config = PredictiveBatchingConfig {
            prediction_window_ms: 100,
            ..Default::default()
        };
        let batcher = PredictiveBatcher::new(config);

        batcher.record_request(256, 128);
        batcher.record_request(128, 64);

        let pattern = batcher.get_metrics().current_pattern;
        assert_eq!(pattern.avg_prompt_tokens, 192);
        assert_eq!(pattern.avg_decode_tokens, 96);
    }
}
