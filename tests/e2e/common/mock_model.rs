// tests/e2e/common/mock_model.rs
//! Deterministic mock model for E2E tests

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Mock model with deterministic behavior
pub struct MockModel {
    failure_sequence: Vec<bool>,
    failure_index: AtomicU64,
    latency_ms: u64,
}

impl MockModel {
    pub fn builder() -> MockModelBuilder {
        MockModelBuilder::default()
    }

    pub async fn forward(&self, _input: &[u64]) -> Result<Vec<u64>, String> {
        // Simulate latency
        if self.latency_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.latency_ms)).await;
        }

        // Check if should fail
        let index = self.failure_index.fetch_add(1, Ordering::Relaxed) as usize;
        if let Some(&should_fail) = self.failure_sequence.get(index) {
            if should_fail {
                return Err("Simulated failure".to_string());
            }
        }

        // Return mock output
        Ok(vec![1u64, 2, 3])
    }
}

/// Builder for MockModel
pub struct MockModelBuilder {
    failure_rate: f64,
    failure_sequence: Option<Vec<bool>>,
    latency_ms: u64,
}

impl Default for MockModelBuilder {
    fn default() -> Self {
        Self {
            failure_rate: 0.0,
            failure_sequence: None,
            latency_ms: 0,
        }
    }
}

impl MockModelBuilder {
    #[allow(dead_code)]
    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate;
        self
    }

    pub fn with_failure_sequence(mut self, sequence: Vec<bool>) -> Self {
        self.failure_sequence = Some(sequence);
        self
    }

    pub fn with_latency_ms(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    pub fn build(self) -> MockModel {
        let failure_sequence = if let Some(seq) = self.failure_sequence {
            seq
        } else {
            vec![false; 1000]
        };

        MockModel {
            failure_sequence,
            failure_index: AtomicU64::new(0),
            latency_ms: self.latency_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model_builder() {
        let mock = MockModel::builder()
            .with_failure_rate(0.1)
            .with_latency_ms(10)
            .build();
        assert_eq!(mock.latency_ms, 10);
    }

    #[tokio::test]
    async fn test_mock_model_forward() {
        let mock = MockModel::builder().build();
        let result = mock.forward(&[1, 2, 3]).await;
        assert!(result.is_ok());
    }
}
