//! OpenAI-API wire types: request bodies, response bodies, streaming SSE chunks, and the OpenAI-specific batch endpoint types.
//!
//! These mirror the public `OpenAI` Chat Completions / Completions / Embeddings
//! / Batch schemas 1:1. Field names and JSON casing match the upstream spec;
//! renaming a field here is a breaking API change.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use vllm_traits::TokenId;

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

/// OpenAI tool type discriminator (P33 v0.x wire-type follow-up).
///
/// Per OpenAI chat-completions spec, the `type` field on a tool
/// definition must currently be `"function"`. Future OpenAI API
/// extensions (e.g. retrieval, code-interpreter) would add more
/// variants; vllm-lite declares only the `Function` variant today
/// because that's all OpenAI ships as of 2026. Future variants
/// would be rejected at the serde layer (axum returns
/// `422 Unprocessable Entity` for unknown enum variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    /// Standard function-calling tool. The associated
    /// `FunctionDefinition` carries the function name, description,
    /// and JSON-Schema `parameters`.
    Function,
}

/// OpenAI `tools[].function` definition (P33 v0.x wire-type follow-up).
///
/// Per OpenAI chat-completions spec: a function definition carries
/// a required `name` plus optional `description` and `parameters`
/// (JSON Schema describing the function's arguments).
///
/// `parameters` is a raw `serde_json::Value` rather than a typed
/// JSON-Schema struct because:
/// - OpenAI's spec is intentionally permissive about JSON Schema
///   drafts (draft-04 through 2020-12 are all accepted server-side).
/// - Modelling the full JSON Schema grammar would balloon the type
///   definition (hundreds of variants for `$ref`, `oneOf`,
///   `allOf`, `anyOf`, etc.) without giving vllm-lite any new
///   capability — the engine doesn't process the schema today
///   (grammar-constrained decoding is v32+ work).
/// - Callers can round-trip the schema verbatim via JSON and
///   `serde_json::Value` preserves byte-for-byte fidelity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name. Used by the model to select which tool to
    /// invoke and by `ToolChoice::Specific` to force a specific
    /// function call. Per OpenAI spec, must match the regex
    /// `^[a-zA-Z0-9_-]{1,64}$` — we do NOT enforce this today
    /// (validator is permissive by design; honoring is v32+ work
    /// so there's no engine-side caller to break).
    pub name: String,
    /// Optional human-readable description. The model uses this to
    /// decide when to invoke the function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional JSON Schema for the function's arguments. Stored
    /// as `serde_json::Value` to preserve fidelity (see the
    /// `FunctionDefinition` doc-comment for the rationale).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// OpenAI `tools[]` entry (P33 v0.x wire-type follow-up).
///
/// Per OpenAI chat-completions spec: each tool carries a `type`
/// discriminator (currently only `"function"`) and a `function`
/// payload. Future tool kinds (retrieval, code-interpreter, etc.)
/// would add new `ToolType` variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    /// Tool-type discriminator. Currently only `Function` is
    /// supported (see the `ToolType` doc-comment).
    #[serde(rename = "type")]
    pub kind: ToolType,
    /// Function-specific payload (name + description + parameters).
    pub function: FunctionDefinition,
}

/// OpenAI `tool_choice` mode discriminator (P33 v0.x wire-type follow-up).
///
/// Per OpenAI chat-completions spec the `tool_choice` parameter
/// can be a string mode (`"none"` / `"auto"` / `"required"`) OR
/// an object that names a specific tool. The string mode is
/// captured by `ToolChoiceMode`; the object form by
/// [`ToolChoice::Specific`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoiceMode {
    /// The model must not call any tool. Equivalent to omitting
    /// `tools` from the request (when paired with `tools = None`).
    #[serde(rename = "none")]
    None,
    /// The model can choose between generating a message or
    /// calling one or more tools. The default mode when `tools`
    /// is non-empty and `tool_choice` is `None`.
    #[serde(rename = "auto")]
    Auto,
    /// The model must call one or more tools. Per OpenAI spec, at
    /// least one tool must be defined in `tools`; the validator
    /// rejects the combination when `tools` is empty / `None`.
    #[serde(rename = "required")]
    Required,
}

/// OpenAI `tool_choice` object variant (P33 v0.x wire-type follow-up).
///
/// Per OpenAI chat-completions spec: the object form of
/// `tool_choice` forces the model to call a specific named tool.
/// The shape is `{"type": "function", "function": {"name": "..."}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceSpecific {
    /// Type discriminator. Currently only `"function"` is supported
    /// (mirrors [`ToolType`]).
    #[serde(rename = "type")]
    pub kind: ToolType,
    /// The specific function to invoke.
    pub function: ToolChoiceFunctionRef,
}

/// OpenAI `tool_choice.function` reference (P33 v0.x wire-type follow-up).
///
/// Carries only the function name; OpenAI doesn't accept a full
/// function definition in `tool_choice` (the full definition lives
/// in `tools[]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoiceFunctionRef {
    /// Name of the function to force-call. Must match the `name`
    /// field of one of the entries in `ChatRequest::tools`.
    pub name: String,
}

/// OpenAI `tool_choice` parameter (P33 v0.x wire-type follow-up).
///
/// `untagged` because the wire shape is "string OR object" —
/// serde matches the first variant that parses. Order matters:
/// the string modes (`None` / `Auto` / `Required`) are tried
/// before the object form (`Specific`) so a bare `"auto"` string
/// doesn't accidentally deserialize as
/// `Specific { kind: Auto, ... }`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String mode (`"none"` / `"auto"` / `"required"`).
    Mode(ToolChoiceMode),
    /// Object form (`{"type": "function", "function": {"name": "..."}}`).
    Specific(ToolChoiceSpecific),
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
    /// **Honoring is end-to-end** for the full OpenAI range
    /// (P27 declaration + P29 sign-aware engine refactor). The
    /// chat handler maps `frequency_penalty` to the engine's
    /// existing `SamplingParams::repeat_penalty` via
    /// `repeat_penalty = (1.0 + frequency_penalty).max(1e-3)`. The
    /// `apply_repeat_penalty` step in
    /// `vllm_core::sampling::sample_batch_with_params` (added by
    /// ARCH-02; sign-aware refactor in P29) handles positive and
    /// negative logits symmetrically:
    ///   - logit >= 0: divide by `repeat_penalty`
    ///   - logit < 0: multiply by `repeat_penalty`
    /// This gives the OpenAI-spec behaviour for both
    /// `frequency_penalty >= 0` (penalize: positive logits shrink,
    /// negative logits grow more negative) and `frequency_penalty < 0`
    /// (boost: positive logits grow, negative logits grow less
    /// negative). The 1e-3 floor prevents divide-by-zero for
    /// extreme negative `frequency_penalty` values (e.g. -1.0 →
    /// `repeat_penalty = 0.0`); values floored to 1e-3 produce
    /// maximum boost, which is the practical limit for the divisor
    /// formulation.
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `frequency_penalty = ?req.frequency_penalty` so the
    /// request's penalty settings are observable in trace logs.
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
    /// OpenAI logit bias (v0.3 wire-type follow-up). A map of token
    /// IDs to additive bias values in the `[-100, 100]` range.
    /// Positive values *increase* the probability of the biased
    /// tokens; negative values *decrease* it.
    ///
    /// Per OpenAI spec the bias values are constrained to the
    /// `[-100, 100]` range and must be finite; the validator on
    /// `validate_chat_request_fields` rejects NaN / ±infinity /
    /// out-of-range values with `400 invalid_request_error`. Any
    /// token ID is accepted (out-of-vocab IDs are silently
    /// ignored by the engine — matches OpenAI's server behaviour).
    ///
    /// **Honoring is end-to-end** via the new
    /// `vllm_core::sampling::apply_logit_bias` helper (added by
    /// P30): the chat handler forwards the map verbatim to
    /// `SamplingParams::logit_bias`, and the sampler adds each
    /// bias to the logit at the corresponding token position
    /// before the temperature / top-k / top-p pipeline. The map
    /// iteration order is non-deterministic (HashMap) but the
    /// *final logits* are deterministic because each bias is
    /// additive and independent per token — so determinism is
    /// preserved.
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `logit_bias_len = ?req.logit_bias.as_ref().map(|m| m.len())`
    /// (count only, not the full map, to keep log lines bounded
    /// for typical maps of up to ~300 entries).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<TokenId, f32>>,
    /// OpenAI logprobs flag (v0.3 wire-type follow-up). Per OpenAI
    /// chat-completions spec: `logprobs: bool` indicates whether to
    /// return the log probability of the sampled token. Default is
    /// `false` (no logprobs returned).
    ///
    /// **Honoring is a no-op** today — the engine's
    /// `sample_batch_with_params` returns only the sampled token;
    /// changing the return type to include logprobs is a wire-
    /// breaking change for the engine boundary. Documented as
    /// v32+ work. The wire-type contract is locked in now so the
    /// declaration-only PR doesn't regress to "rejected by serde"
    /// for callers who already send the field.
    ///
    /// Validated by `validate_chat_logprobs` (no range check on
    /// the bool itself; cross-field rule: `top_logprobs=Some` requires
    /// `logprobs=true`).
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `logprobs = ?req.logprobs` for parity with P21/P22/
    /// P23/P27/P28/P29/P30 observability plumbing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// OpenAI top-logprobs count (v0.3 wire-type follow-up). Per
    /// OpenAI chat-completions spec: an integer in the range
    /// `0..=20` specifying how many of the most likely tokens to
    /// return log probabilities for at each position. Only
    /// meaningful when `logprobs = true`; the validator rejects
    /// `top_logprobs = Some` with `logprobs = false`.
    ///
    /// **Honoring is a no-op** today (same rationale as
    /// [`ChatRequest::logprobs`] — engine-side top-K logprob
    /// generation is v32+ work).
    ///
    /// Validated by `validate_chat_logprobs` (range `0..=20`,
    /// cross-field rule: requires `logprobs=true`).
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `top_logprobs = ?req.top_logprobs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// OpenAI `tools[]` (P33 v0.x wire-type follow-up —
    /// declaration + validation). Per OpenAI chat-completions spec:
    /// a list of tools the model may call. Currently only
    /// function-calling tools are supported (see [`ToolType`]); the
    /// `function.parameters` field is a JSON Schema stored as
    /// `serde_json::Value` to preserve fidelity (see
    /// [`FunctionDefinition`] for the rationale).
    ///
    /// **Honoring is a no-op** today — tool calling requires a
    /// grammar-constrained decoder (JSON-schema → grammar) and a
    /// per-request tool schema cache. Architecture-level work,
    /// tracked as v32+. The wire-type contract is locked in now so
    /// the declaration-only PR doesn't regress to "rejected by
    /// serde" for callers who already send the field.
    ///
    /// Validated by `validate_chat_tool_choice` (cross-field rule:
    /// `tool_choice = Some(Required)` or `tool_choice = Some(Specific)`
    /// requires `tools` to be non-empty; `tool_choice =
    /// Some(Specific{ function: { name: X } })` requires `tools`
    /// to contain a function named `X`). Field names +
    /// descriptions + parameters are *not* validated today —
    /// honoring is v32+ work so there's no engine-side caller to
    /// break.
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `tools_len = ?req.tools.as_ref().map(|v| v.len())`
    /// (count only, not the full array, to keep log lines bounded
    /// for typical tool lists of up to ~10 entries).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// OpenAI `tool_choice` (P33 v0.x wire-type follow-up —
    /// declaration + validation). Per OpenAI chat-completions spec:
    /// either a mode string (`"none"` / `"auto"` / `"required"`)
    /// or an object (`{"type": "function", "function": {"name":
    /// "..."}}`) that forces the model to call a specific tool.
    /// Modeled via the [`ToolChoice`] enum with `#[serde(untagged)]`
    /// so both forms round-trip 1:1.
    ///
    /// **Honoring is a no-op** today — same rationale as
    /// [`ChatRequest::tools`]. Engine-side tool-calling is
    /// v32+ work (grammar-constrained decoder + per-request tool
    /// schema cache).
    ///
    /// Validated by `validate_chat_tool_choice` (see the
    /// `tools` field doc-comment for the rule set). Field names
    /// are *not* validated against the regex `^[a-zA-Z0-9_-]{1,64}$`
    /// today — the validator is permissive by design until
    /// honoring is wired.
    ///
    /// Threaded into the chat handler's `tracing::info!(...)` log
    /// lines as `tool_choice = ?req.tool_choice` (Debug-printable
    /// because `ToolChoice` derives `Debug`; the field is small
    /// enough to log in full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Per-token logprob information rendered in chat completion responses
/// when the request set `logprobs = true` (P36 v0.3 wire-type follow-up
/// engine wire-through).
///
/// Mirrors the OpenAI spec's `choices[].logprobs.content[]` shape 1:1.
/// `token` is the decoded string form (UTF-8 round-tripped from the
/// sampled `TokenId`). `logprob` is the natural-log probability under
/// the actual sampling distribution (post-filter: after
/// `repeat_penalty`, `presence_penalty`, `logit_bias`, temperature
/// scaling, `top_k`, and `top_p` nucleus cutoff). `bytes` is the
/// UTF-8 byte representation of `token` (used by tokenizers that
/// emit byte-level BPE markers; we emit the actual UTF-8 bytes of
/// the decoded string per OpenAI's reference implementation).
///
/// `top_logprobs` is populated when the request also set
/// `top_logprobs = Some(n)` — it contains the top-`n` most-likely
/// alternative tokens at this position alongside their logprobs,
/// sorted by logprob descending. `None` when the request did not ask
/// for top-K logprobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLogprob {
    /// Decoded token string.
    pub token: String,
    /// `ln(P(token))` under the actual sampling distribution.
    pub logprob: f32,
    /// UTF-8 byte representation of `token` (matches OpenAI's
    /// reference serializer).
    pub bytes: Option<Vec<u8>>,
    /// Top-K alternative `(token, logprob)` pairs sorted by
    /// `logprob` descending. `None` when the request did not ask
    /// for top-K logprobs; `Some(vec![])` when asked but the
    /// sampler produced zero finite-probability tokens (e.g. all
    /// logits were `-inf`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<ChatLogprob>>,
}

/// Container for `ChatChoice::logprobs` (P36 v0.3 wire-type follow-up
/// engine wire-through). `None` when the request did not ask for
/// logprobs; `Some(container)` when `logprobs = true` was requested.
///
/// `content` is parallel to `ChatChoice::message.content` — one
/// entry per generated token, in decoding order. The chat endpoint
/// does not include the prompt tokens (those would appear in a
/// separate `prompt_logprobs` field that OpenAI's chat API
/// doesn't currently expose).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoiceLogprobs {
    /// Per-generated-token logprob entries (parallel to the
    /// assistant message's token sequence).
    pub content: Vec<ChatLogprob>,
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
    /// Per-token logprob information (P36 v0.3 wire-type follow-up
    /// engine wire-through). `None` unless the request set
    /// `logprobs = true`. When the request set
    /// `logprobs = true` but `top_logprobs = Some(0)` (or omitted),
    /// `content` carries one entry per generated token with only
    /// the sampled token's logprob (no `top_logprobs` sub-field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatChoiceLogprobs>,
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
    /// Per-token logprob information for this chunk (P36 v0.3
    /// wire-type follow-up engine wire-through). `None` unless
    /// the request set `logprobs = true`. Each intermediate chunk
    /// carries exactly one entry in `content` (the token emitted
    /// on that chunk); the final chunk (the one with
    /// `finish_reason`) carries zero entries because the
    /// finish-reason chunk emits no token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatChoiceLogprobs>,
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
    /// `repeat_penalty = (1.0 + frequency_penalty).max(1e-3)` (P29
    /// sign-aware refactor). Negative values are honored end-to-end
    /// for boost semantic; the 1e-3 floor prevents divide-by-zero
    /// for extreme negative values.
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
    /// OpenAI logit bias (v0.3 wire-type follow-up). See
    /// [`ChatRequest::logit_bias`] for the full contract:
    /// per OpenAI spec the valid range is `[-100, 100]` (default
    /// no bias); the validator on
    /// `validate_completion_request_fields` rejects NaN /
    /// ±infinity / out-of-range values with `400`. Honoring is
    /// end-to-end via the new
    /// `vllm_core::sampling::apply_logit_bias` helper (added by
    /// P30) — the completions handler forwards the map verbatim
    /// to `SamplingParams::logit_bias`. The completions handler
    /// does not currently log the field (parity with the
    /// `seed` / `user` / `frequency_penalty` / `presence_penalty`
    /// fields — chat handler logs them, completions handler does
    /// not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<TokenId, f32>>,
    /// OpenAI logprobs count (v0.3 wire-type follow-up). Per OpenAI
    /// legacy-completions spec: an integer in the range `0..=5`
    /// specifying how many of the most likely tokens to return
    /// log probabilities for at each position. The completions
    /// endpoint's `logprobs` has a *different* type than the chat
    /// endpoint's `logprobs` (int 0-5 here vs bool on chat) per
    /// the OpenAI spec — P31 declares both with the correct types
    /// rather than unifying behind a common representation.
    ///
    /// **Honoring is a no-op** today — same rationale as
    /// [`ChatRequest::logprobs`]: engine-side top-K logprob
    /// generation is v32+ work.
    ///
    /// Validated by `validate_completion_logprobs` (range `0..=5`).
    /// The completions handler does not currently log the field
    /// (parity with `seed` / `user` / `frequency_penalty` /
    /// `presence_penalty` / `logit_bias` rationale).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    /// OpenAI `echo` flag (P32 v0.x wire-type follow-up — declaration
    /// + validation). Per OpenAI legacy-completions spec: when `true`,
    /// the response echoes the prompt back as a prefix to the
    /// generated continuation in the `text` field (instead of just
    /// returning the continuation). Default `false`.
    ///
    /// **Honoring is a no-op** today — engine would need to
    /// prepend the prompt to `CompletionChoice.text` in the
    /// streaming + non-streaming paths. Tracked as v32+ work
    /// (mechanical, but adds a tokenizer dependency to the
    /// response side). The wire-type contract is locked in now so
    /// the declaration-only PR doesn't regress to "rejected by
    /// serde" for callers who already send the field.
    ///
    /// Validated by `validate_completion_meta` (cross-field rule:
    /// `echo = true` cannot coexist with `best_of > 1` per OpenAI
    /// spec — best_of is meaningless when echoing because the
    /// server picks the single highest-mean-logprob completion
    /// without exposing logprobs, so the user has no way to
    /// disambiguate which of N completions was selected).
    ///
    /// The completions handler does not currently log the field
    /// (parity with the `seed` / `user` / `frequency_penalty` /
    /// `presence_penalty` / `logit_bias` rationale — completions
    /// handler accepts these at the wire type but does not log any
    /// of them).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// OpenAI `suffix` (P32 v0.x wire-type follow-up — declaration +
    /// validation). Per OpenAI legacy-completions spec: a string that
    /// comes after the inserted completion. Useful for code-completion
    /// UIs that pre-fill the suffix (e.g. the closing `}` of a
    /// function body) and want the model to fill only the gap.
    /// Default `None`.
    ///
    /// **Honoring is a no-op** today — engine would need to append
    /// the suffix to `CompletionChoice.text` in the response. Tracked
    /// as v32+ work (mechanical). The wire-type contract is locked
    /// in now so the declaration-only PR doesn't regress to "rejected
    /// by serde" for callers who already send the field.
    ///
    /// Validated by `validate_completion_meta` (no range / length
    /// check per OpenAI spec — any string is accepted). The
    /// completions handler does not currently log the field (parity
    /// with the rationale above).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// OpenAI `best_of` count (P32 v0.x wire-type follow-up —
    /// declaration + validation). Per OpenAI legacy-completions spec:
    /// an integer ≥ 1 specifying how many completions to generate
    /// server-side, returning the "best" one (highest mean log
    /// probability over the generated tokens). Default `1`.
    ///
    /// **Honoring is a no-op** today — engine would need to sample
    /// `best_of` times (with the same prompt + sampling params),
    /// rank by mean logprob, and return the single best. The
    /// logprob-ranking primitive requires the same v32+ engine work
    /// as the `logprobs` field (P31), so the two are co-dependent.
    /// Tracked as v32+ work.
    ///
    /// Validated by `validate_completion_meta` (`>= 1` per OpenAI
    /// spec, cross-field rule: `best_of > 1` cannot coexist with
    /// `echo = true` because the user has no way to disambiguate
    /// which of N completions was returned; also `best_of > 1`
    /// cannot coexist with `n > 1` because each "best" is one of N
    /// total — but `n > 1` is already rejected by the existing
    /// validator). The completions handler does not currently log
    /// the field (parity with the rationale above).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
}

/// Per-token logprob information rendered in legacy text-completion
/// responses when the request set `logprobs > 0` (P36 v0.3 wire-type
/// follow-up engine wire-through).
///
/// Mirrors the OpenAI spec's `choices[].logprobs.top_logprobs[]`
/// shape 1:1. `token` is the decoded string form, `logprob` is the
/// natural-log probability under the actual sampling distribution
/// (post-filter), `bytes` is the UTF-8 byte representation of
/// `token`. Length is bounded by `SamplingParams::top_logprobs` (≤
/// `ChatRequest::top_logprobs.unwrap_or(0)` for chat, ≤
/// `CompletionRequest::logprobs.unwrap_or(0)` for legacy
/// completions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionLogprob {
    /// Decoded token string.
    pub token: String,
    /// `ln(P(token))` under the actual sampling distribution.
    pub logprob: f32,
    /// UTF-8 byte representation of `token` (matches OpenAI's
    /// reference serializer).
    pub bytes: Option<Vec<u8>>,
}

/// Container for `CompletionChoice::logprobs` (P36 v0.3 wire-type
/// follow-up engine wire-through). `None` when the request did not
/// ask for logprobs; `Some(container)` when `logprobs > 0` was
/// requested.
///
/// `tokens` / `token_logprobs` / `top_logprobs` are all parallel
/// arrays of length `N` (the number of generated tokens). `tokens[i]`
/// is the decoded string form of the `i`-th generated token;
/// `token_logprobs[i]` is its logprob under the actual sampling
/// distribution; `top_logprobs[i]` is the list of top-K alternatives
/// at that position (each entry is a [`CompletionLogprob`]). When
/// the request set `logprobs = Some(k)` with `k > 0`, each
/// `top_logprobs[i]` has up to `k` entries; when `logprobs = Some(0)`,
/// each `top_logprobs[i]` is an empty list (sampled-token logprobs
/// are still emitted via `token_logprobs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoiceLogprobs {
    /// Decoded token strings (parallel to `token_logprobs` /
    /// `top_logprobs`).
    pub tokens: Vec<String>,
    /// Per-generated-token logprobs under the actual sampling
    /// distribution. Length matches `tokens`.
    pub token_logprobs: Vec<f32>,
    /// Per-generated-token top-K alternative logprob lists. Length
    /// matches `tokens`; each inner list has at most `logprobs`
    /// entries (≤ 5 per OpenAI spec).
    pub top_logprobs: Vec<Vec<CompletionLogprob>>,
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
    /// Per-token logprob information (P36 v0.3 wire-type follow-up
    /// engine wire-through). `None` unless the request set
    /// `logprobs = Some(n)` with `n > 0`. When the request set
    /// `logprobs = Some(0)`, the container is `Some(empty)` —
    /// sampling-emitted but no top-K alternatives — matching
    /// OpenAI's behavior of including `tokens: []` /
    /// `token_logprobs: []` even when `logprobs = 0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionChoiceLogprobs>,
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
