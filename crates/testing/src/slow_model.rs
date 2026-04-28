//! SlowModel - Mock model with configurable delay
//!
//! Useful for testing timeout behavior, backpressure, and scheduling.

use std::thread;
use std::time::Duration;
use vllm_traits::{BatchOutput, ModelBackend, Result, SeqId, TokenId};

/// Model that introduces a configurable delay for deterministic testing.
///
/// Useful for:
/// - Testing timeout behavior
/// - Verifying backpressure mechanisms
/// - Simulating slow inference
///
/// # Example
///
/// ```rust,ignore
/// use vllm_testing::SlowModel;
///
/// let mut model = SlowModel::new(Duration::from_millis(10));
/// // Each forward() call will take at least 10ms
/// ```
#[derive(Debug, Clone)]
pub struct SlowModel {
    delay: Duration,
    return_token: TokenId,
}

impl SlowModel {
    /// Create a new SlowModel with the specified delay
    pub fn new(delay: Duration) -> Self {
        Self {
            delay,
            return_token: 1,
        }
    }

    /// Create a new SlowModel with delay and custom return token
    pub fn with_token(delay: Duration, return_token: TokenId) -> Self {
        Self {
            delay,
            return_token,
        }
    }

    /// Set the delay duration
    pub fn delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Set the return token
    pub fn return_token(mut self, token: TokenId) -> Self {
        self.return_token = token;
        self
    }
}

impl ModelBackend for SlowModel {
    fn forward(
        &mut self,
        seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<BatchOutput> {
        thread::sleep(self.delay);
        Ok(BatchOutput {
            seq_ids: seq_ids.to_vec(),
            next_tokens: seq_ids.iter().map(|_| self.return_token).collect(),
        })
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> Result<Vec<Vec<f32>>> {
        thread::sleep(self.delay);
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn embed(
        &mut self,
        input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        thread::sleep(self.delay);
        Ok(input_tokens
            .iter()
            .map(|t| t.iter().map(|_| 0.0).collect())
            .collect())
    }

    fn vocab_size(&self) -> usize {
        151936
    }

    fn num_layers(&self) -> usize {
        32
    }

    fn num_heads(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slow_model_delay() {
        let model = SlowModel::new(Duration::from_millis(5));
        assert_eq!(model.delay, Duration::from_millis(5));
    }

    #[test]
    fn test_slow_model_with_token() {
        let model = SlowModel::with_token(Duration::from_millis(10), 42);
        assert_eq!(model.return_token, 42);
    }

    #[test]
    fn test_slow_model_builder() {
        let model = SlowModel::new(Duration::from_secs(1)).return_token(100);

        assert_eq!(model.delay, Duration::from_secs(1));
        assert_eq!(model.return_token, 100);
    }

    #[test]
    fn test_slow_model_forward_takes_time() {
        use std::time::Instant;

        let mut model = SlowModel::new(Duration::from_millis(50));
        let start = Instant::now();

        let output = model
            .forward(&[1], &[vec![1]], &[vec![0]], &[vec![0]], &[0], &[true])
            .unwrap();

        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(50));
        assert_eq!(output.next_tokens, vec![1]);
    }
}
