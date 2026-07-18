//! OpenAI-API wire types: request bodies, response bodies, streaming SSE chunks, and the OpenAI-specific batch endpoint types.
//!
//! These mirror the public `OpenAI` Chat Completions / Completions / Embeddings
//! / Batch schemas 1:1. Field names and JSON casing match the upstream spec;
//! renaming a field here is a breaking API change.
use serde::{Deserialize, Serialize};

use crate::util::time::unix_now_secs;

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
    /// Construct a [`Usage`] from raw `usize` counts, saturating to `0`
    /// on platforms where `usize > i64::MAX` (essentially never on
    /// 64-bit targets, but defensive for portability).
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
    /// Construct an [`ErrorResponse`] with no machine-readable `code`.
    /// Use [`ErrorResponse::with_code`] when the failure category maps
    /// to a stable identifier the client can branch on.
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

    /// Construct an [`ErrorResponse`] with a machine-readable `code` field.
    ///
    /// `OpenAI`'s error spec defines a `code` slot for stable identifiers such as
    /// `"context_length_exceeded"` or `"model_not_found"`. Use this when you
    /// know the specific failure category; clients can switch on `code` to
    /// drive retry / fallback logic.
    #[must_use]
    pub fn with_code(message: &str, error_type: &str, code: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: Some(code.to_string()),
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

/// Output format selector (OpenAI `response_format`).
///
/// Mirrors the OpenAI `response_format` field on `/v1/chat/completions`.
/// Serializes as `{"type": "text"}` / `{"type": "json_object"}` via the
/// `tag = "type"` attribute so the JSON shape matches upstream 1:1.
///
/// **v0.2 scope (P22)**: only the `Text` and `JsonObject` variants are
/// accepted. The third OpenAI variant, `{type: "json_schema", ...}`,
/// requires a grammar-constrained decoder and is explicitly deferred to
/// v0.3 (see `docs/reference/openai-compatibility.md` v0.2 follow-ups).
/// `serde` will reject a `json_schema` payload with a 400-class
/// deserialization error; `validate_chat_response_format` provides an
/// extra safety net for clients that send `{"type": "json_schema"}`
/// with extra fields serde might tolerate.
///
/// **Honoring is a no-op today.** The engine's sampler does not enforce
/// JSON syntax; `ResponseFormat::JsonObject` is accepted as a no-op
/// pass-through (same as `Text` for the sampler). v0.3 + v32+ work
/// (per the OpenAI compat matrix) adds a constrained-decoding hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Default plain-text generation. Equivalent to omitting
    /// `response_format` entirely.
    Text,
    /// JSON-object mode. The model is asked to produce valid JSON,
    /// but vllm-lite does not currently constrain the sampler to
    /// enforce JSON syntax. See the doc-comment above for the
    /// v0.3 / v32+ constrained-decoding roadmap.
    JsonObject,
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
    /// Optional end-user identifier for safety / abuse tracking (P21 v0.2
    /// wire-type declaration; honored as a tracing pass-through only —
    /// vllm-lite has no auth/persistence layer that consumes it today).
    /// Per OpenAI spec there is no format/length validation; any string
    /// is accepted. Downstream consumers (rate-limiter, audit log) can
    /// subscribe to the structured `tracing` field without changing the
    /// HTTP boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Output format selector (P22 v0.2 declaration; v0.3 / v32+
    /// constrained-decoding roadmap). Only `Text` and `JsonObject`
    /// are accepted today — `JsonSchema` requires a grammar-
    /// constrained decoder and is deferred to v0.3. Honoring is a
    /// no-op (see [`ResponseFormat`] doc-comment).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// RNG seed for "best effort determinism" (P23 v0.2 wire-type
    /// declaration). Per the OpenAI spec, if `seed` is set the system
    /// will "best effort to sample deterministically" — same seed +
    /// same model + same prompt should produce the same output. We
    /// accept any `i64` per OpenAI spec (no range / sign validation).
    ///
    /// **Honoring is a no-op today** — vllm-lite's sampler reads from
    /// `rand`'s thread-local RNG which is currently unseeded. The
    /// value flows through the existing `tracing::info!(...)` log
    /// lines as `seed = ?req.seed` so determinism is at least
    /// observable in trace logs (e.g. for diagnosing "did the client
    /// set a seed or not?"). Engine-side RNG seeding is v32+ work;
    /// the wire-type contract is locked in now so the
    /// declaration-only PR doesn't regress to "rejected by serde".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// OpenAI frequency penalty (v0.3 wire-type follow-up). Per the
    /// OpenAI API spec the valid range is `[-2.0, 2.0]`: positive
    /// values penalise tokens that have already appeared in the
    /// response (more occurrences → larger penalty), negative values
    /// *encourage* repetition. The default is `0` (no penalty).
    ///
    /// **Honoring is end-to-end** for non-negative values:
    /// the chat handler maps `frequency_penalty` to the engine's
    /// existing `SamplingParams::repeat_penalty` via
    /// `repeat_penalty = max(1.0, 1.0 + frequency_penalty)`. The
    /// `apply_repeat_penalty` step in
    /// `vllm_core::sampling::sample_batch_with_params` (added by
    /// ARCH-02) divides logits at previously-seen token positions by
    /// `repeat_penalty`, which matches OpenAI's "halve logit on each
    /// repetition" semantics for `frequency_penalty >= 0`.
    ///
    /// **Negative values are clamped to `1.0` (no penalty)** because
    /// `repeat_penalty < 1.0` would invert the engine's logit-divide
    /// math (boosting repetition) — that is the desired OpenAI
    /// behaviour but the current `apply_repeat_penalty` flips the
    /// sign of negative logits when dividing by a value `< 1.0`,
    /// producing undefined ordering rather than a clean boost.
    /// Surfacing the v0.3 / v32+ "boost" semantics requires either a
    /// new `apply_repeat_boost` helper or a sign-aware refactor of
    /// `apply_repeat_penalty`; either is mechanical but out of scope
    /// for the declaration PR. The validator still rejects
    /// `frequency_penalty < -2.0` (per OpenAI spec) so callers learn
    /// about truly out-of-range values; the in-range negative case
    /// silently degrades to "no penalty" rather than producing
    /// garbage output.
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `frequency_penalty = ?req.frequency_penalty` so the
    /// request's penalty settings are observable in trace logs even
    /// when honoring is partially clamped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// OpenAI presence penalty (v0.3 wire-type follow-up). Per the
    /// OpenAI API spec the valid range is `[-2.0, 2.0]`: positive
    /// values penalise tokens that have appeared at all (binary
    /// "seen?" check), negative values *encourage* presence of
    /// already-seen tokens. The default is `0` (no penalty).
    ///
    /// **Honoring is end-to-end** via the new
    /// `vllm_core::sampling::apply_presence_penalty` helper (added
    /// by P28): the chat handler forwards `presence_penalty` to the
    /// engine's `SamplingParams::presence_penalty` slot, and the
    /// helper subtracts the penalty from the logit of every
    /// *distinct* seen token regardless of occurrence count —
    /// matching OpenAI's presence-style semantic. Positive values
    /// discourage repetition; negative values *encourage*
    /// repetition (because subtracting a negative is the same as
    /// adding). Unlike `frequency_penalty` (clamped via `max(1.0,
    /// 1.0 + value)` to work around the `apply_repeat_penalty`
    /// logit-divide sign-flip bug for negative values),
    /// `presence_penalty` is an *additive* bias so the value is
    /// forwarded verbatim — no clamping needed.
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `presence_penalty = ?req.presence_penalty` for
    /// parity with P21/P22/P23/P27 observability plumbing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
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
    /// Construct a non-streaming [`ChatResponse`]. Stamps the `object`
    /// slot as `"chat.completion"` and `created` to the current Unix
    /// second; streaming callers should build [`ChatChunk`]s directly.
    #[must_use]
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: unix_now_secs(),
            model,
            choices,
            usage,
        }
    }
}

/// A single choice inside an SSE [`ChatChunk`]. The `delta` carries
/// only the partial message text emitted on this chunk — typically
/// the `role` on the first chunk and `content` on subsequent chunks,
/// mirroring the `OpenAI` streaming protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index (0-based; constant across stream chunks).
    pub index: i32,
    /// Streaming delta — partial message, usually only `role` on first chunk and `content` on subsequent chunks.
    pub delta: ChatMessage,
    /// Set on the final chunk; `None` on intermediate deltas.
    pub finish_reason: Option<String>,
}

/// A single chunk in a chat-completion SSE stream. The server emits
/// one `ChatChunk` per generated token, followed by a final chunk
/// with `finish_reason = Some("stop")` and the OpenAI sentinel
/// `"[DONE]"` appended to the SSE payload.
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
    /// Construct a streaming [`ChatChunk`] for the given single choice.
    /// Stamps `object = "chat.completion.chunk"` and `created` to the
    /// current Unix second.
    #[must_use]
    pub fn new(id: String, model: String, choice: ChatChunkChoice) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: unix_now_secs(),
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
    /// Nucleus sampling cumulative probability cutoff.
    pub top_p: Option<f32>,
    /// Maximum generated tokens.
    pub max_tokens: Option<i64>,
    /// Enable streaming response.
    pub stream: Option<bool>,
    /// Number of independent completions.
    pub n: Option<i64>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
    /// Optional end-user identifier for safety / abuse tracking (P21 v0.2
    /// wire-type declaration; honored as a tracing pass-through only).
    /// See [`ChatRequest::user`] for the full contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// RNG seed for "best effort determinism" (P23 v0.2 wire-type
    /// declaration). Same contract as [`ChatRequest::seed`]: per OpenAI
    /// spec any `i64` is accepted; honoring is a no-op today (the
    /// sampler is unseeded). The completions handler does not currently
    /// log the field — adding a new `tracing::info!` line is deferred
    /// to keep parity with the `user` field (the chat handler logs
    /// both, the completions handler accepts both at the wire type but
    /// does not log either).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// OpenAI frequency penalty (v0.3 wire-type follow-up). See
    /// [`ChatRequest::frequency_penalty`] for the full contract:
    /// per OpenAI spec the valid range is `[-2.0, 2.0]` (default
    /// `0`); the completions handler maps the field to the engine's
    /// `SamplingParams::repeat_penalty` via
    /// `repeat_penalty = max(1.0, 1.0 + frequency_penalty)` so
    /// non-negative values are honored end-to-end. Negative values
    /// are silently clamped to `1.0` (no penalty) for the same
    /// reason documented on [`ChatRequest::frequency_penalty`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// OpenAI presence penalty (v0.3 wire-type follow-up). See
    /// [`ChatRequest::presence_penalty`] for the full contract:
    /// per OpenAI spec the valid range is `[-2.0, 2.0]` (default
    /// `0`); honoring is end-to-end via the new
    /// `vllm_core::sampling::apply_presence_penalty` helper (added
    /// by P28). The completions handler forwards the value verbatim
    /// to `SamplingParams::presence_penalty`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

/// A single choice in a text-completion response. The `text` field
/// holds the raw continuation (no chat-template rendering) for the
/// legacy `/v1/completions` endpoint.
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
    /// Construct a [`CompletionResponse`] for the legacy
    /// `/v1/completions` endpoint. Stamps `object = "text_completion"`
    /// and `created` to the current Unix second.
    #[must_use]
    pub fn new(id: String, model: String, choices: Vec<CompletionChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: unix_now_secs(),
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

/// Deprecated alias for [`Embedding`]. Retained for backward
/// compatibility with clients written against the 0.19.x wire format.
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
    /// Build an [`EmbeddingsResponse`] from raw dense vectors. The
    /// `usage.prompt_tokens` field is set to the total dimension count
    /// across all embeddings; `completion_tokens` is always `0` since
    /// embeddings have no autoregressive output.
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
