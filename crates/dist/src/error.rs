//! Error types for the `vllm-dist` crate.
//!
//! This module serves as the **canonical home** for error types emitted by
//! distributed primitives (`tensor_parallel`, `distributed_kv`, `pipeline`,
//! `grpc`). Each submodule's error type is re-exported here so callers have a
//! single, stable import path: `vllm_dist::error::*`.
//!
//! # Canonical Import Paths
//!
//! | Error type             | Canonical path                             |
//! |------------------------|--------------------------------------------|
//! | Tensor parallel errors | `vllm_dist::error::TensorParallelError`    |
//! | Pipeline errors        | `vllm_dist::error::PipelineError`          |
//! | gRPC server errors     | `vllm_dist::error::GrpcError`              |
//!
//! # Why `TensorParallelError` Definition Lives in `vllm-traits`
//!
//! The `TensorParallelError` enum is semantically owned by `vllm-dist`
//! (it describes tensor-parallel primitives). However, the **technical**
//! definition lives in `vllm-traits` because:
//!
//! - The dependency direction is `vllm-dist → vllm-traits` (not the reverse).
//!   Inverting this would create a cycle.
//! - `vllm-traits` is **never feature-gated**, while `vllm-dist` is only
//!   compiled when the `multi-node` feature is enabled. Error types that
//!   appear in non-multi-node public APIs (e.g., `Result<_, TensorParallelError>`
//!   return types in trait signatures) must be available without the feature.
//! - `vllm-traits` is the project's stable-types crate: error enums that
//!   cross feature boundaries belong there.
//!
//! ## Resolution
//!
//! This module re-exports `vllm_traits::TensorParallelError` so consumers
//! importing `vllm_dist::error::TensorParallelError` get the canonical type
//! without needing to know about the underlying home. The `vllm-traits`
//! re-export remains as a backward-compat shim and is the technical source.
//!
//! See: `.planning/audit/architecture/REPORT.md` (v19.0 finding ARCH-F-13),
//! Phase 31 plan `31-05` for the resolution discussion.

/// `TensorParallelError` — errors emitted by tensor-parallel primitives.
///
/// Re-exported from `vllm_traits::TensorParallelError` (the technical
/// definition). New code should prefer importing from this canonical path.
pub use vllm_traits::TensorParallelError;

pub use crate::pipeline::PipelineError;

/// `GrpcError` — errors emitted by the gRPC server bootstrap path.
///
/// Replaces the previous `Box<dyn std::error::Error>` return type of
/// `start_grpc_server` so callers can match on typed variants instead of
/// stringifying. `#[source]` chains preserve the underlying cause for
/// log correlation.
#[derive(Debug, thiserror::Error)]
pub enum GrpcError {
    #[error("failed to bind gRPC listener")]
    Bind(#[source] std::io::Error),
    #[error("gRPC transport error")]
    Transport(#[source] tonic::transport::Error),
}

impl From<std::io::Error> for GrpcError {
    fn from(e: std::io::Error) -> Self {
        Self::Bind(e)
    }
}

impl From<tonic::transport::Error> for GrpcError {
    fn from(e: tonic::transport::Error) -> Self {
        Self::Transport(e)
    }
}
