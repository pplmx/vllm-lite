use std::sync::atomic::{AtomicU64, Ordering};

/// Type of metric being recorded
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    Counter(String),
    Gauge(String),
    Histogram(String),
}

impl MetricType {
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Counter(name) | Self::Gauge(name) | Self::Histogram(name) => name,
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
    /// Construct a zero-initialized counter.
    #[must_use]
    pub const fn new_counter() -> Self {
        Self::Counter(AtomicU64::new(0))
    }

    /// Construct a zero-initialized gauge.
    #[must_use]
    pub const fn new_gauge() -> Self {
        Self::Gauge(AtomicU64::new(0))
    }

    /// Add `delta` to the value. No-op when this is a Gauge or Histogram —
    /// use [`Self::set`] for gauges instead.
    pub fn increment(&self, delta: u64) {
        if let Self::Counter(c) = self {
            c.fetch_add(delta, Ordering::Relaxed);
        }
    }

    /// Overwrite the value. No-op when this is a Counter or Histogram —
    /// use [`Self::increment`] for counters instead.
    pub fn set(&self, value: u64) {
        if let Self::Gauge(g) = self {
            g.store(value, Ordering::Relaxed);
        }
    }

    /// Read the current value as a `u64`. Returns 0 for Histogram variants
    /// (use dedicated histogram accessors for those).
    pub fn as_u64(&self) -> u64 {
        match self {
            Self::Counter(c) => c.load(Ordering::Relaxed),
            Self::Gauge(g) => g.load(Ordering::Relaxed),
            Self::Histogram(_) => 0,
        }
    }
}

/// Labels for metric dimensions
#[derive(Debug, Clone, Default)]
pub struct MetricLabels {
    labels: Vec<(String, String)>,
}

impl MetricLabels {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }

    #[must_use]
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
