//! OpenAI-API compatible surface: chat completions, completions, embeddings, models, and the async batch endpoint.
//!
//! Sub-routers mounted under `/v1` from the top-level `api.rs`.
//! Streaming responses use SSE (server-sent events) for chat + completions.
/// Async batch API endpoint (create/get/list batches).
pub mod batch;
/// Chat completions (streaming + non-streaming) with conversational state.
pub mod chat;
/// Jinja2 chat-template resolution per model architecture.
pub mod chat_template;
/// Text completions (legacy `/v1/completions` style).
pub mod completions;
/// Embedding generation.
pub mod embeddings;
/// Model metadata (`/v1/models`).
pub mod models;
/// OpenAI parameter validation and normalization.
pub mod sampling_validation;
/// Wire-format types shared across OpenAI endpoints.
pub mod types;
