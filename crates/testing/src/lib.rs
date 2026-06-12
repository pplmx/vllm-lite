//! Test Infrastructure for vllm-lite
//!
//! Provides unified test utilities including:
//! - TestHarness for test environment setup
//! - Mock models with deterministic behavior
//! - Request factory for generating test requests

pub mod builders;
pub mod fixtures;
pub mod harness;
pub mod mocks;
pub mod request_factory;
pub mod slow_model;
pub mod utils;

pub use builders::{BatchBuilder, RequestBuilder};
pub use fixtures::TestFixtures;
pub use harness::TestHarness;
pub use mocks::{ConstModel, FakeModel, IncrementModel, NeverProgressModel, StubModel};
pub use request_factory::RequestFactory;
pub use slow_model::SlowModel;
pub use utils::{assert_batch_consistency, create_simple_batch, generate_random_tokens};

pub mod prelude {
    pub use super::{
        BatchBuilder, ConstModel, FakeModel, IncrementModel, NeverProgressModel, RequestBuilder,
        RequestFactory, SlowModel, StubModel, TestFixtures, TestHarness, assert_batch_consistency,
        create_simple_batch, generate_random_tokens,
    };
}
