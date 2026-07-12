//! OpenAI-API compatible surface: chat completions, completions, embeddings, models, and the async batch endpoint.
//!
//! Sub-routers mounted under `/v1` from the top-level `api.rs`.
//! Streaming responses use SSE (server-sent events) for chat + completions.
pub mod batch;
pub mod chat;
pub mod chat_template;
pub mod completions;
pub mod embeddings;
pub mod models;
pub mod sampling_validation;
pub mod types;
