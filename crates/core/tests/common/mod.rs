//! Re-export testing utilities from vllm-testing crate.
//!
//! This module provides backward compatibility for tests that import
//! from `super::common`. New code should use `vllm-testing` directly.

pub use vllm_testing::{ConstModel, FakeModel, IncrementModel, NeverProgressModel, StubModel};

pub use vllm_testing::builders::{BatchBuilder, RequestBuilder};

pub use vllm_testing::fixtures::TestFixtures;

pub use vllm_testing::utils::{
    assert_batch_consistency, create_simple_batch, generate_random_tokens,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_re_exports() {
        let _model = IncrementModel;
        let _builder = RequestBuilder::new(1);
        let _batch_builder = BatchBuilder::new();
        let _fixtures = TestFixtures::default_scheduler_config();
    }
}
