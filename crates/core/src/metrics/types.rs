use std::sync::atomic::{AtomicU64, Ordering};

/// Type of metric being recorded
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    Counter(String),
    Gauge(String),
    Histogram(String),
}

impl MetricType {
    pub fn name(&self) -> &str {
        match self {
            MetricType::Counter(name) => name,
            MetricType::Gauge(name) => name,
            MetricType::Histogram(name) => name,
        }
    }
}

/// Value stored for a metric
#[derive(Debug)]
pub enum MetricValue {
    Counter(AtomicU64),
    Gauge(AtomicU64),
    Histogram(Vec<u64>),
}

impl MetricValue {
    pub fn new_counter() -> Self {
        MetricValue::Counter(AtomicU64::new(0))
    }

    pub fn new_gauge() -> Self {
        MetricValue::Gauge(AtomicU64::new(0))
    }

    pub fn increment(&self, delta: u64) {
        if let MetricValue::Counter(c) = self {
            c.fetch_add(delta, Ordering::Relaxed);
        }
    }

    pub fn set(&self, value: u64) {
        if let MetricValue::Gauge(g) = self {
            g.store(value, Ordering::Relaxed);
        }
    }

    pub fn as_u64(&self) -> u64 {
        match self {
            MetricValue::Counter(c) => c.load(Ordering::Relaxed),
            MetricValue::Gauge(g) => g.load(Ordering::Relaxed),
            MetricValue::Histogram(_) => 0,
        }
    }
}

/// Labels for metric dimensions
#[derive(Debug, Clone, Default)]
pub struct MetricLabels {
    labels: Vec<(String, String)>,
}

impl MetricLabels {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }

    pub fn as_slice(&self) -> &[(String, String)] {
        &self.labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_type_creation() {
        let counter = MetricType::Counter("test_counter".to_string());
        assert_eq!(counter.name(), "test_counter");
    }

    #[test]
    fn test_metric_value_increment() {
        let counter = MetricValue::new_counter();
        counter.increment(1);
        assert_eq!(counter.as_u64(), 1);
    }

    #[test]
    fn test_metric_labels() {
        let labels = MetricLabels::new()
            .with("method", "POST")
            .with("endpoint", "/v1/completions");
        assert_eq!(labels.as_slice().len(), 2);
    }
}
