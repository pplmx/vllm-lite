#![allow(clippy::module_name_repetitions)]
//! `SchedulerEngine` — continuous-batching engine.
//!
//! This module is split into four focused sub-modules that share a
//! single `SchedulerEngine` struct (defined in `state`):
//!
//! - `graph` (3 methods) — CUDA Graph helpers (`build_batch_with_graph`
//!   plus the two private helpers it relies on).
//! - `update` — `update`, the post-step state update invoked after
//!   the model forward pass returns.
//! - `memory` (6 methods) — preemption, KV cache rollback, request
//!   cancellation, pressure reporting, KV usage stats, and the
//!   prefix cache accessor.
//! - `state` — the struct itself, its constructor, the major
//!   lifecycle methods (`add_request`, `build_batch`, `schedule`,
//!   `set_policy`), the `Default` impl, and the read-only state
//!   accessors.
//!
//! All `impl SchedulerEngine` blocks across the sub-modules contribute
//! methods to the same type, so callers see a single flat API.

pub mod graph;
pub mod memory;
pub mod state;
pub mod update;

pub use state::SchedulerEngine;

// Unit tests are extracted to `tests.rs` to keep this file under the
// 800-line soft cap. See `tests.rs` for the test surface
// (add_request / build_batch / update / phase / multiple-requests /
// memory-pressure / prefix-cache / metrics).
#[cfg(test)]
mod tests;
