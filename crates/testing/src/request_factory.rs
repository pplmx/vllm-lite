//! RequestFactory - Test request generator
//!
//! Provides a configurable factory for creating test requests
//! with various token patterns and sampling settings.

use vllm_core::types::{Priority, Request, SamplingParams};

/// Configuration for generating test requests
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub min_max_tokens: usize,
    pub max_max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            min_tokens: 64,
            max_tokens: 512,
            min_max_tokens: 8,
            max_max_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

impl RequestConfig {
    pub fn min_tokens(mut self, n: usize) -> Self {
        self.min_tokens = n;
        self
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn min_max_tokens(mut self, n: usize) -> Self {
        self.min_max_tokens = n;
        self
    }

    pub fn max_max_tokens(mut self, n: usize) -> Self {
        self.max_max_tokens = n;
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

/// Factory for generating test requests
///
/// Provides a convenient way to create test requests with
/// various configurations for integration tests.
///
/// # Example
///
/// ```rust,ignore
/// use vllm_testing::RequestFactory;
///
/// let factory = RequestFactory::new()
///     .min_tokens(64)
///     .max_tokens(512)
///     .max_max_tokens(32);
///
/// // Create a single request
/// let request = factory.create(1);
///
/// // Create multiple requests
/// let requests = factory.create_batch(10);
/// ```
#[derive(Debug, Clone)]
pub struct RequestFactory {
    config: RequestConfig,
    counter: u64,
}

impl RequestFactory {
    /// Create a new RequestFactory with default configuration
    pub fn new() -> Self {
        Self {
            config: RequestConfig::default(),
            counter: 1,
        }
    }

    /// Create a RequestFactory from a custom configuration
    pub fn from_config(config: RequestConfig) -> Self {
        Self { config, counter: 1 }
    }

    /// Set the minimum prompt token count
    pub fn min_tokens(mut self, n: usize) -> Self {
        self.config.min_tokens = n;
        self
    }

    /// Set the maximum prompt token count
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.config.max_tokens = n;
        self
    }

    /// Set the minimum max_tokens for generation
    pub fn min_max_tokens(mut self, n: usize) -> Self {
        self.config.min_max_tokens = n;
        self
    }

    /// Set the maximum max_tokens for generation
    pub fn max_max_tokens(mut self, n: usize) -> Self {
        self.config.max_max_tokens = n;
        self
    }

    /// Set the temperature for sampling
    pub fn temperature(mut self, t: f32) -> Self {
        self.config.temperature = t;
        self
    }

    /// Create a single request with the next sequence ID
    pub fn create(&mut self) -> Request {
        self.create_with_id(self.counter)
    }

    /// Create a request with a specific sequence ID
    pub fn create_with_id(&mut self, seq_id: u64) -> Request {
        let prompt_tokens = self.generate_tokens();
        let max_tokens = self.generate_max_tokens();

        let request = Request::new(seq_id, prompt_tokens, max_tokens);
        self.counter += 1;
        request
    }

    /// Create a batch of requests with sequential IDs
    pub fn create_batch(&mut self, count: usize) -> Vec<Request> {
        (0..count).map(|_| self.create()).collect()
    }

    /// Create a batch of requests starting from a specific ID
    pub fn create_batch_from(&mut self, start_id: u64, count: usize) -> Vec<Request> {
        (0..count)
            .map(|i| self.create_with_id(start_id + i as u64))
            .collect()
    }

    /// Create a request with specific prompt tokens
    pub fn create_with_prompt(&mut self, prompt: Vec<u32>, max_tokens: usize) -> Request {
        let params = SamplingParams {
            temperature: self.config.temperature,
            top_k: self.config.top_k,
            top_p: self.config.top_p,
            ..Default::default()
        };

        let id = self.counter;
        self.counter += 1;

        Request {
            id,
            prompt,
            max_tokens,
            sampling_params: params,
            priority: Priority::default(),
        }
    }

    fn generate_tokens(&self) -> Vec<u32> {
        let count = rand_token_count(self.config.min_tokens, self.config.max_tokens);
        (0..count).map(|_| rand::random::<u32>() % 32000).collect()
    }

    fn generate_max_tokens(&self) -> usize {
        rand_token_count(self.config.min_max_tokens, self.config.max_max_tokens)
    }

    /// Reset the internal counter
    pub fn reset_counter(&mut self) {
        self.counter = 1;
    }
}

impl Default for RequestFactory {
    fn default() -> Self {
        Self::new()
    }
}

fn rand_token_count(min: usize, max: usize) -> usize {
    if min >= max {
        return min;
    }
    let range = max - min;
    min + (rand::random::<u32>() as usize % range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_factory_default() {
        let factory = RequestFactory::new();
        assert_eq!(factory.config.min_tokens, 64);
        assert_eq!(factory.config.max_tokens, 512);
    }

    #[test]
    fn test_request_factory_custom_config() {
        let factory = RequestFactory::new()
            .min_tokens(32)
            .max_tokens(256)
            .max_max_tokens(64);

        assert_eq!(factory.config.min_tokens, 32);
        assert_eq!(factory.config.max_tokens, 256);
        assert_eq!(factory.config.max_max_tokens, 64);
    }

    #[test]
    fn test_create_request() {
        let mut factory = RequestFactory::new();
        let request = factory.create();

        assert_eq!(request.id, 1);
        assert!(!request.prompt.is_empty());
        assert!(request.max_tokens > 0);
    }

    #[test]
    fn test_create_batch() {
        let mut factory = RequestFactory::new();
        let requests = factory.create_batch(5);

        assert_eq!(requests.len(), 5);
        assert_eq!(requests[0].id, 1);
        assert_eq!(requests[4].id, 5);
    }

    #[test]
    fn test_create_with_prompt() {
        let mut factory = RequestFactory::new();
        let request = factory.create_with_prompt(vec![1, 2, 3, 4, 5], 20);

        assert_eq!(request.prompt, vec![1, 2, 3, 4, 5]);
        assert_eq!(request.max_tokens, 20);
    }

    #[test]
    fn test_counter_increments() {
        let mut factory = RequestFactory::new();
        factory.create();
        factory.create();
        let third = factory.create();

        assert_eq!(third.id, 3);
    }

    #[test]
    fn test_reset_counter() {
        let mut factory = RequestFactory::new();
        factory.create();
        factory.create();
        factory.reset_counter();
        let request = factory.create();

        assert_eq!(request.id, 1);
    }
}
