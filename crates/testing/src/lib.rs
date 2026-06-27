//! Test Infrastructure for vllm-lite
//!
//! Provides unified test utilities including:
//! - `TestHarness` for test environment setup
//! - Mock models with deterministic behavior (`FakeModel`, `ConstModel`, etc.)
//! - `RequestFactory` for generating test requests
//! - Shared fixtures (`TestFixtures`, builders, batch helpers)
//!
//! # Architecture Decision: Why No `vllm-testkit` + `vllm-harness` Lemon Pair Split
//!
//! During the v19.0 architecture audit (Phase 20), this crate was evaluated for
//! splitting into a "lemon pair" — `vllm-testkit` (lightweight utilities) and
//! `vllm-harness` (full environment setup) — per the standard two-crate pattern.
//! After analysis (Phase 31 / ML-05), the decision is **NOT to split**. Rationale:
//!
//! 1. **No consumer asymmetry.** All callers (unit tests in every crate +
//!    integration tests in `crates/*/tests/`) consume the same exports
//!    (`TestHarness`, `RequestFactory`, mock models, builders). There is no
//!    "unit-only" or "integration-only" subset.
//!
//! 2. **Compile-time benefit is negligible.** The largest module is
//!    `request_factory.rs` (~287 LOC) and `harness.rs` (~220 LOC). A split
//!    would force consumers to add a second `dev-dependencies` entry without
//!    reducing the transitive closure — both crates would still depend on
//!    `vllm-traits` and `vllm-core`.
//!
//! 3. **Tight coupling.** `RequestFactory` depends on `BatchBuilder` (both in
//!    this crate). `TestHarness` consumes `RequestFactory`. A split would force
//!    `vllm-harness → vllm-testkit` dependency direction, but `TestHarness` is
//!    the heaviest module — splitting puts the larger dep "downstream" of the
//!    smaller, inverting normal layering.
//!
//! 4. **Maintenance cost exceeds benefit.** Two crates doubles the
//!    `Cargo.toml`, `lib.rs`, `prelude`, and CI surface area for a 702 LOC
//!    crate. Phase 28 doc coverage push (97.8%) shows the single-crate model
//!    scales fine.
//!
//! ## Re-evaluation Triggers
//!
//! Split if any of these become true:
//! - A consumer emerges that needs only the request factory (no harness) AND
//!   the savings exceed the split overhead (rough threshold: >5 distinct
//!   "testkit-only" callers).
//! - The crate exceeds ~3000 LOC and individual module boundaries harden.
//! - Compile times for `cargo test -p <crate>` (single-crate tests) become
//!   dominated by `vllm-testing` rebuild cost.
//!
//! See: `.planning/audit/architecture/REPORT.md` (v19.0 audit),
//! `.planning/audit/SYNTHESIS.md` (Theme: test architecture),
//! Phase 31 plan `31-04` for the original evaluation.

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

// Curated top-level re-exports of the most-used test utilities.
// Modules remain accessible via direct path (`vllm_testing::builders::*`)
// for less-common exports that don't warrant top-level ergonomics.
pub use fixtures::TestFixtures;
pub use harness::TestHarness;
pub use mocks::{ConstModel, FakeModel, IncrementModel, StubModel};
pub use request_factory::RequestFactory;
pub use slow_model::SlowModel;

/// prelude: prelude module.
///
/// Re-exports the commonly-used test utilities for `use vllm_testing::prelude::*;`.
/// Excludes `SlowModel` (heavyweight; only needed for #[ignore] benchmark tests).
pub mod prelude {
    pub use super::{
        ConstModel, FakeModel, IncrementModel, RequestFactory, StubModel, TestFixtures, TestHarness,
    };
}
