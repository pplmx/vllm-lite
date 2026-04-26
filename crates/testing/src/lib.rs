//! Test Infrastructure for vllm-lite
//!
//! Provides unified test utilities including:
//! - TestHarness for test environment setup
//! - Mock models with deterministic behavior
//! - Request factory for generating test requests

pub mod harness;
pub mod mocks;
pub mod request_factory;
pub mod slow_model;

pub use harness::TestHarness;
pub use mocks::{ConstModel, FakeModel, IncrementModel, NeverProgressModel, StubModel};
pub use request_factory::RequestFactory;
pub use slow_model::SlowModel;

pub mod prelude {
    pub use super::{
        ConstModel, FakeModel, IncrementModel, NeverProgressModel, RequestFactory, SlowModel,
        StubModel, TestHarness,
    };
}
