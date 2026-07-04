//! OpenAI Batch API: file-upload style async job submission with 24h SLA. Wire types in `types.rs`, lifecycle in `manager.rs`, axum handlers in `handler.rs`.
#![allow(clippy::module_name_repetitions)]
pub mod handler;
pub mod manager;
pub mod types;

pub use manager::BatchManager;
pub use types::{BatchEndpoint, BatchResponse};
