//! OpenAI-API wire types: request bodies, response bodies, streaming SSE chunks, and the OpenAI-specific batch endpoint types.
//!
//! These mirror the public OpenAI Chat Completions / Completions / Embeddings
//! / Batch schemas 1:1. Field names and JSON casing match the upstream spec;
//! renaming a field here is a breaking API change.
use serde::{Deserialize, Serialize};

/// Token usage statistics for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: i64,
    /// Tokens generated in the completion.
    pub completion_tokens: i64,
    /// `prompt_tokens + completion_tokens` (caller may validate against this total).
    pub total_tokens: i64,
}

impl Usage {
    #[must_use]
    pub fn new(prompt: usize, completion: usize) -> Self {
        let prompt = i64::try_from(prompt).unwrap_or(0);
        let completion = i64::try_from(completion).unwrap_or(0);
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// Error details following `OpenAI` API error format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Human-readable error message.
    pub message: String,
    /// Error category (`"invalid_request_error"`, `"server_error"`, etc.).
    #[serde(rename = "type")]
    pub error_type: String,
    /// Optional machine-readable error code (e.g. `"context_length_exceeded"`).
    pub code: Option<String>,
}

/// Error response wrapper following `OpenAI` API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// The error detail payload.
    pub error: ErrorDetail,
}

impl ErrorResponse {
    #[must_use]
    pub fn new(message: &str, error_type: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: None,
            },
        }
    }
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Message text content.
    pub content: String,
    /// Optional author name (rare; supported for multi-user logs).
    pub name: Option<String>,
}

/// Request body for chat completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model identifier (e.g. `"qwen3-4b"`).
    pub model: String,
    /// Ordered conversation history.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature (`0.0`–`2.0`); `None` = model default.
    pub temperature: Option<f32>,
    /// Nucleus sampling cumulative probability cutoff.
    pub top_p: Option<f32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: Option<i64>,
    /// When `true`, stream via server-sent events.
    pub stream: Option<bool>,
    /// Number of independent completions to generate.
    pub n: Option<i64>,
    /// Stop sequences; generation halts when any is emitted.
    pub stop: Option<Vec<String>>,
}

/// A choice in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index (0-based; matches `ChatRequest::n`).
    pub index: i32,
    /// The generated assistant message.
    pub message: ChatMessage,
    /// `"stop"`, `"length"`, or `"tool_calls"` (when the model invokes a tool).
    pub finish_reason: Option<String>,
}

/// Response from chat completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Unique completion identifier (`"chatcmpl-..."`).
    pub id: String,
    /// Always `"chat.completion"` for non-streaming, `"chat.completion.chunk"` for streaming.
    pub object: String,
    /// Unix timestamp at which the response was generated.
    pub created: i64,
    /// Echo of the requested model id.
    pub model: String,
    /// Generated completions (length matches `ChatRequest::n`).
    pub choices: Vec<ChatChoice>,
    /// Token accounting for this response.
    pub usage: Usage,
}

impl ChatResponse {
    #[must_use]
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
            model,
            choices,
            usage,
        }
    }
}

/// `ChatChunkChoice`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index (0-based; constant across stream chunks).
    pub index: i32,
    /// Streaming delta — partial message, usually only `role` on first chunk and `content` on subsequent chunks.
    pub delta: ChatMessage,
    /// Set on the final chunk; `None` on intermediate deltas.
    pub finish_reason: Option<String>,
}

/// `ChatChunk`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    /// Stream identifier (shared across all chunks in the same response).
    pub id: String,
    /// Always `"chat.completion.chunk"` for streaming.
    pub object: String,
    /// Unix timestamp at the start of the stream.
    pub created: i64,
    /// Echo of the requested model id.
    pub model: String,
    /// Streaming choices (typically one per request).
    pub choices: Vec<ChatChunkChoice>,
}

impl ChatChunk {
    #[must_use]
    pub fn new(id: String, model: String, choice: ChatChunkChoice) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
            model,
            choices: vec![choice],
        }
    }
}

/// Request body for text completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model id; optional for the legacy `/v1/completions` endpoint.
    pub model: Option<String>,
    /// Raw prompt text (no chat-template applied).
    pub prompt: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum generated tokens.
    pub max_tokens: Option<i64>,
    /// Enable streaming response.
    pub stream: Option<bool>,
    /// Number of independent completions.
    pub n: Option<i64>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
}

/// `CompletionChoice`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Generated continuation text.
    pub text: String,
    /// Choice index (0-based).
    pub index: i32,
    /// Termination reason.
    pub finish_reason: Option<String>,
}

/// Response from text completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Unique completion identifier (`"cmpl-..."`).
    pub id: String,
    /// Always `"text_completion"`.
    pub object: String,
    /// Unix timestamp at which the response was generated.
    pub created: i64,
    /// Echo of the requested model id.
    pub model: String,
    /// Generated completions.
    pub choices: Vec<CompletionChoice>,
    /// Token accounting for this response.
    pub usage: Usage,
}

impl CompletionResponse {
    #[must_use]
    pub fn new(id: String, model: String, choices: Vec<CompletionChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
            model,
            choices,
            usage,
        }
    }
}

/// Request body for embeddings endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    /// Model id of the embedding model.
    pub model: String,
    /// Input texts to embed (batch endpoint accepts strings).
    pub input: Vec<String>,
}

/// Embedding: single embedding item in an embeddings response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Always `"embedding"`.
    pub object: String,
    /// Dense vector representation.
    pub embedding: Vec<f32>,
    /// Index of this embedding within the input batch.
    pub index: i32,
}

/// Deprecated alias for [`Embedding`].
#[deprecated(since = "0.20.0", note = "use Embedding instead")]
pub type EmbeddingData = Embedding;

/// Response from embeddings endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    /// Always `"list"`.
    pub object: String,
    /// Embedding results (one per input string).
    pub data: Vec<Embedding>,
    /// Echo of the requested model id.
    pub model: String,
    /// Token accounting for this response.
    pub usage: Usage,
}

impl EmbeddingsResponse {
    #[must_use]
    pub fn new(embeddings: Vec<Vec<f32>>, model: String) -> Self {
        let items: Vec<Embedding> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, e)| Embedding {
                object: "embedding".to_string(),
                embedding: e,
                index: i32::try_from(i).unwrap_or(0),
            })
            .collect();

        let total_tokens: i64 = items
            .iter()
            .map(|d| i64::try_from(d.embedding.len()).unwrap_or(0))
            .sum();

        Self {
            object: "list".to_string(),
            data: items,
            model,
            usage: Usage::new(usize::try_from(total_tokens).unwrap_or(0), 0),
        }
    }
}
