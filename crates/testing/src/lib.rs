//! Test Infrastructure for vllm-lite
//!
//! Provides unified test utilities including:
//! - TestHarness for test environment setup
//! - Mock models with deterministic behavior
//! - Request factory for generating test requests

/// builders: builders module.
pub mod builders;
/// fixtures: fixtures module.
pub mod fixtures;
/// harness: harness module.
pub mod harness;
/// mocks: mocks module.
pub mod mocks;
/// request_factory: request factory module.
pub mod request_factory;
/// slow_model: slow model module.
pub mod slow_model;
/// utils: utils module.
pub mod utils;

pub use builders::{BatchBuilder, RequestBuilder};
pub use fixtures::TestFixtures;
pub use harness::TestHarness;
pub use mocks::{ConstModel, FakeModel, IncrementModel, NeverProgressModel, StubModel};
pub use request_factory::RequestFactory;
pub use slow_model::SlowModel;
pub use utils::{assert_batch_consistency, create_simple_batch, generate_random_tokens};

/// prelude: prelude module.
pub mod prelude {
    pub use super::{
        BatchBuilder, ConstModel, FakeModel, IncrementModel, NeverProgressModel, RequestBuilder,
        RequestFactory, SlowModel, StubModel, TestFixtures, TestHarness, assert_batch_consistency,
        create_simple_batch, generate_random_tokens,
    };
}
