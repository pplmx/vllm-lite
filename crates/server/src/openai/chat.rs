//! `OpenAI` Chat Completions endpoint: `POST /v1/chat/completions`.
//!
//! Handles both unary and SSE-streaming responses. Validates the
//! request against the `OpenAI` schema, tokenises the messages through
//! the chat template, and dispatches to the engine. See
//! `types.rs` for request/response shapes.
#![allow(clippy::module_name_repetitions)]
use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
};
use futures::stream;
use std::convert::Infallible;
use tokio::sync::mpsc;

use super::chat_template::{self, ChatTemplate};
use super::sampling_validation::validate_sampling_params;
use super::types::{
    ChatChoice, ChatChunk, ChatChunkChoice, ChatMessage, ChatRequest, ChatResponse, ErrorResponse,
    Usage,
};
use crate::ApiState;

fn should_skip_token_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> bool {
    text.is_empty() || tokenizer.is_special_token(text)
}

fn clean_completion_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> String {
    tokenizer.clean_special_tokens(text)
}

/// Build a model-ready prompt string from a list of chat `messages`, using
/// the architecture-appropriate [`ChatTemplate`] (`ChatML`, Llama-2, etc.).
///
/// Thin wrapper around `super::chat_template::build_prompt` exposed at the
/// `openai::chat` module path so handlers can use it without naming the
/// template submodule.
#[must_use]
pub fn build_prompt_from_messages(template: ChatTemplate, messages: &[ChatMessage]) -> String {
    chat_template::build_prompt(template, messages)
}

pub(crate) fn validate_chat_request(
    req: &ChatRequest,
) -> Result<(), (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "model is required",
                "invalid_request_error",
            )),
        ));
    }
    if req.messages.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "messages is required",
                "invalid_request_error",
            )),
        ));
    }
    Ok(())
}

async fn handle_chat(
    state: &ApiState,
    req: ChatRequest,
) -> Result<ChatResponse, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let request_id = format!(
        "req_{}",
        uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
    );

    if let Err((status, err_resp)) = validate_chat_request(&req) {
        tracing::warn!(
            request_id = %request_id,
            status = %status,
            error = %err_resp.error.message,
            "Request validation failed"
        );
        return Err((status, err_resp));
    }

    let template = ChatTemplate::for_architecture(state.architecture);
    let prompt = build_prompt_from_messages(template, &req.messages);
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens_len = prompt_tokens.len();

    tracing::info!(
        request_id = %request_id,
        prompt_tokens = prompt_tokens_len,
        "Request started"
    );
    let max_tokens = usize::try_from(req.max_tokens.unwrap_or(100)).unwrap_or(100);
    let total_max = prompt_tokens_len + max_tokens;

    // Production-readiness §4: reject requests whose
    // prompt + max_tokens would exceed the model's context
    // length. Without this gate a 10× oversize prompt
    // exhausts KV blocks before any application-level
    // validation runs. We emit the OpenAI-standard
    // `context_length_exceeded` code so OpenAI-compatible
    // clients can detect the failure mode and split the
    // request.
    if let Some(max_model_len) = state.max_model_len {
        if total_max > max_model_len {
            let message = format!(
                "prompt_tokens ({prompt_tokens_len}) + max_tokens ({max_tokens}) \
                 = {total_max} exceeds the model's context length ({max_model_len})"
            );
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(ErrorResponse::with_code(
                    &message,
                    "invalid_request_error",
                    "context_length_exceeded",
                )),
            ));
        }
    }

    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
    }

    // Reject sampling parameters the engine cannot honour (currently
    // beam_width > 1) BEFORE enqueuing — see `sampling_validation`.
    validate_sampling_params(&request.sampling_params)?;

    let (response_tx, mut response_rx) = mpsc::channel(64);

    // REL-01 (technical due diligence): use `try_send` so a full
    // engine mailbox fails fast with `503 engine_overloaded`
    // instead of blocking on capacity. Distinguishes two failure
    // modes:
    //   - `Full`:   mailbox is saturated → 503 `engine_overloaded`
    //   - `Closed`: engine has shut down → 503 `engine_unavailable`
    state
        .engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
            // Non-streaming handler doesn't need a seq_id
            // round-trip; the request runs to natural completion
            // (or max_tokens) and we drop the oneshot on the floor.
            seq_id_tx: None,
        })
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => engine_overloaded_error(),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => engine_unavailable_error(),
        })?;

    let mut tokens = Vec::new();
    while let Some(token) = response_rx.recv().await {
        tokens.push(token);
    }

    let raw_decode = state.tokenizer.decode(&tokens);

    let completion_text = clean_completion_text(&state.tokenizer, &raw_decode);

    let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let output_tokens_len = tokens.len();

    tracing::info!(
        request_id = %request_id,
        output_tokens = output_tokens_len,
        duration_ms = duration_ms,
        "Request completed"
    );
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: completion_text,
            name: None,
        },
        finish_reason: Some("stop".to_string()),
    };

    let usage = Usage::new(prompt_tokens_len, tokens.len());

    Ok(ChatResponse::new(
        format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        req.model,
        vec![choice],
        usage,
    ))
}

/// OpenAI-compatible `/v1/chat/completions` HTTP handler.
///
/// Dispatches to either the streaming (SSE) or non-streaming path based on
/// `req.stream`. Both paths:
///   1. Validate the request (`validate_chat_request`).
///   2. Render the chat messages into a model-ready prompt using the
///      architecture's [`ChatTemplate`].
///   3. Encode, submit to the engine via `engine_tx`, and await tokens.
///   4. Decode + post-process the tokens back into OpenAI-shaped JSON (or
///      SSE chunks in the streaming case).
///
/// # Errors
///
/// Returns a `(StatusCode, ErrorResponse)` pair when:
///   - request validation fails (`BAD_REQUEST`)
///   - the engine channel is closed (`SERVICE_UNAVAILABLE`, code `engine_unavailable`)
///   - token decoding or response serialization fails
///
/// # Panics
///
/// Panics only if SSE chunk serialization fails (it cannot, given the
/// payload types are plain `serde_json`-derived structs).
pub async fn chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.stream.unwrap_or(false) {
        stream_chat_completion(state, req).await
    } else {
        non_stream_chat_completion(state, req).await
    }
}

/// Streaming (SSE) variant of `/v1/chat/completions`.
///
/// Builds the prompt, submits the request to the engine, and pipes each
/// emitted token through the chat-template tokenizer into an SSE stream
/// of [`ChatChunk`]s. A trailing `[DONE]` sentinel is appended when the
/// engine channel closes.
///
/// # Errors
///
/// Returns `SERVICE_UNAVAILABLE` (code `engine_unavailable`) when the
/// engine channel is closed at submission time. Streaming itself cannot
/// fail mid-flight — the SSE transport is best-effort and a client
/// disconnect simply drops the remaining chunks.
///
/// # Panics
///
/// Panics only if SSE chunk serialization fails (it cannot, given the
/// payload types are plain `serde_json`-derived structs).
// `async` is intentional for symmetry with `non_stream_chat_completion`
// and to leave room for future async work (e.g., metrics collection,
// tracing spans) without another signature change. Currently the
// streaming prep is fully synchronous (`UnboundedSender::send` is
// sync), so clippy flags this — silence here, not in the caller.
#[allow(clippy::unused_async)]
async fn stream_chat_completion(
    state: ApiState,
    req: ChatRequest,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let request_id = format!(
        "req_{}",
        uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
    );
    let template = ChatTemplate::for_architecture(state.architecture);
    let prompt = build_prompt_from_messages(template, &req.messages);
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens_len = prompt_tokens.len();

    tracing::info!(
        request_id = %request_id,
        model = %req.model,
        prompt_tokens = prompt_tokens_len,
        "Streaming request started"
    );

    let max_tokens = usize::try_from(req.max_tokens.unwrap_or(100)).unwrap_or(100);
    let total_max = prompt_tokens.len() + max_tokens;

    // Production-readiness §4: context-length gate (streaming
    // variant). See non_stream_chat_completion for the full
    // rationale; we apply the same check here so SSE clients
    // get the same `400 context_length_exceeded` instead of a
    // hung-up connection that opens, then dies on the first
    // forward pass.
    if let Some(max_model_len) = state.max_model_len {
        if total_max > max_model_len {
            let message = format!(
                "prompt_tokens ({prompt_tokens_len}) + max_tokens ({max_tokens}) \
                 = {total_max} exceeds the model's context length ({max_model_len})"
            );
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(ErrorResponse::with_code(
                    &message,
                    "invalid_request_error",
                    "context_length_exceeded",
                )),
            ));
        }
    }

    let model = req.model.clone();
    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);
    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
    }

    // Reject sampling parameters the engine cannot honour (currently
    // beam_width > 1) BEFORE enqueuing — see `sampling_validation`.
    validate_sampling_params(&request.sampling_params)?;

    // Production-readiness recommendation: allocate a seq_id
    // round-trip oneshot so we can learn the engine-assigned id
    // and forward `EngineMessage::CancelRequest` if the client
    // disconnects mid-stream. Without this, the engine keeps
    // generating tokens for a caller that has already gone away,
    // wasting GPU cycles and KV-block slots.
    let (seq_id_tx, seq_id_rx) = tokio::sync::oneshot::channel();
    let (response_tx, response_rx) = mpsc::channel(64);
    state
        .engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
            seq_id_tx: Some(seq_id_tx),
        })
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => engine_overloaded_error(),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => engine_unavailable_error(),
        })?;

    // Block briefly until the engine assigns the seq_id. The
    // engine's run loop drains messages synchronously in the same
    // step that handles AddRequest, so under normal load this
    // resolves within microseconds; we cap with a short timeout
    // so a stalled engine doesn't block the HTTP handler forever
    // (the engine is on a dedicated thread; if it's wedged, the
    // whole server is wedged — but failing fast at the handler
    // is still better than hanging).
    let seq_id: vllm_traits::SeqId =
        match tokio::time::timeout(std::time::Duration::from_secs(1), seq_id_rx).await {
            Ok(Ok(id)) => id,
            Ok(Err(_)) => return Err(engine_unavailable_error()),
            Err(_) => return Err(engine_unavailable_error()),
        };
    // If the engine rejected admission (empty prompt), `seq_id`
    // is the sentinel 0. We still stream — the engine closes the
    // response_tx promptly — but the cancel guard below will not
    // send CancelRequest for sentinel 0 since `cancel_request`
    // returns false on unknown ids anyway.

    // Drop guard: when axum drops this Sse stream (client
    // disconnect, connection error, server shutdown), send
    // CancelRequest so the engine releases KV blocks. We store
    // the seq_id in an Atomic so the guard can read it without
    // holding a &mut into the inner stream state.
    let cancel_guard = std::sync::Arc::new(CancelOnDrop {
        engine_tx: state.engine_tx.clone(),
        seq_id: std::sync::atomic::AtomicU64::new(seq_id),
        fired: std::sync::atomic::AtomicBool::new(false),
        request_id: request_id.clone(),
    });

    let tokenizer = state.tokenizer;
    let stream = stream::unfold(
        (response_rx, cancel_guard.clone(), false),
        move |(mut rx, cancel_guard, mut done)| {
            let tokenizer = tokenizer.clone();
            let model = model.clone();
            let request_id = request_id.clone();
            let start = start;
            async move {
                // After the final [DONE] event was emitted, end
                // the stream on the next poll so axum closes the
                // HTTP body. Clients that watch for end-of-stream
                // (e.g. `content-length`, half-close) need this
                // to know the request is finished.
                if done {
                    return None;
                }
                if let Some(token) = rx.recv().await {
                    let text = tokenizer.decode(&[token]);
                    if should_skip_token_text(&tokenizer, &text) {
                        return Some((
                            Ok::<Event, Infallible>(Event::default().data("")),
                            (rx, cancel_guard, done),
                        ));
                    }
                    let chunk = ChatChunk::new(
                        "chatcmpl-stream".to_string(),
                        model.clone(),
                        ChatChunkChoice {
                            index: 0,
                            delta: ChatMessage {
                                role: "assistant".to_string(),
                                content: text,
                                name: None,
                            },
                            finish_reason: None,
                        },
                    );
                    let sse_payload =
                        // invariant: serializing a known-good struct (plain serde_json types);
                        // to_string cannot fail.
                        serde_json::to_string(&chunk).expect("Failed to serialize chat chunk");
                    Some((
                        Ok(Event::default().data(sse_payload)),
                        (rx, cancel_guard, done),
                    ))
                } else {
                    // Channel closed - could be normal completion or client disconnect.
                    // Mark the guard as fired so its Drop impl doesn't
                    // send a redundant CancelRequest for a sequence
                    // that finished naturally.
                    cancel_guard.disarm();
                    let chunk = ChatChunk::new(
                        "chatcmpl-stream".to_string(),
                        model.clone(),
                        ChatChunkChoice {
                            index: 0,
                            delta: ChatMessage {
                                role: "assistant".to_string(),
                                content: String::new(),
                                name: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        },
                    );
                    let sse_payload =
                        // invariant: serializing a known-good struct (plain serde_json types);
                        // to_string cannot fail.
                        serde_json::to_string(&chunk).expect("Failed to serialize chat chunk");
                    tracing::info!(
                        request_id = %request_id,
                        duration_ms = %u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
                        "Streaming request completed"
                    );
                    // Final [DONE] event. Stream terminates on
                    // the NEXT call: `done` is set, so the
                    // unfold closure returns None next time
                    // axum polls it. This ensures the SSE body
                    // closes after the [DONE] marker — clients
                    // relying on `content-length` or
                    // connection-close to detect end-of-stream
                    // would otherwise hang.
                    done = true;
                    Some((
                        Ok(Event::default().data(format!("{sse_payload}\n\n[DONE]"))),
                        (rx, cancel_guard, done),
                    ))
                }
            }
        },
    );

    // The Sse stream holds the only strong reference to
    // cancel_guard at this point. When the client disconnects,
    // axum drops the Sse, the unfold state (which keeps a clone)
    // is dropped, and the LAST remaining cancel_guard is
    // dropped — firing CancelRequest. The natural-completion
    // path explicitly disarms the guard before returning the
    // final [DONE] event so we don't double-cancel.
    Ok(Sse::new(Box::pin(stream)).into_response())
}

/// Drop guard that sends `EngineMessage::CancelRequest` for its
/// captured seq_id when the last strong reference is dropped,
/// unless [`CancelOnDrop::disarm`] has been called.
///
/// Production-readiness recommendation: when an SSE client
/// disconnects mid-stream, axum drops the response stream;
/// without a guard, the engine keeps generating tokens for a
/// caller that has already gone away, wasting GPU cycles and
/// holding KV blocks until natural completion or max_tokens.
pub(crate) struct CancelOnDrop {
    pub(crate) engine_tx: crate::api::EngineHandle,
    pub(crate) seq_id: std::sync::atomic::AtomicU64,
    pub(crate) fired: std::sync::atomic::AtomicBool,
    pub(crate) request_id: String,
}

impl CancelOnDrop {
    /// Mark this guard as a no-op. Call this once the stream has
    /// reached its natural completion (the engine closed
    /// `response_tx` and we sent `[DONE]`), so the Drop impl
    /// doesn't issue a redundant `CancelRequest`.
    pub(crate) fn disarm(&self) {
        self.fired.store(true, std::sync::atomic::Ordering::Release);
    }

    fn seq_id(&self) -> vllm_traits::SeqId {
        self.seq_id.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.fired.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let seq_id = self.seq_id();
        if seq_id == 0 {
            // Sentinel from a rejected admission (e.g. empty prompt).
            // The engine never admitted the sequence, so there is
            // nothing to cancel.
            return;
        }
        // Best-effort: `try_send` because the engine mailbox may
        // already be closed during shutdown. Failing here just
        // means the engine was torn down first, which is fine.
        let _ = self
            .engine_tx
            .try_send(vllm_core::types::EngineMessage::CancelRequest { seq_id });
        tracing::info!(
            request_id = %self.request_id,
            seq_id,
            "Streaming client disconnected; engine sequence cancelled"
        );
    }
}

/// Non-streaming variant of `/v1/chat/completions`.
///
/// Delegates to [`handle_chat`] (which validates the request, builds
/// the prompt, collects all tokens, and assembles a [`ChatResponse`])
/// and wraps the result in a JSON HTTP response.
///
/// # Errors
///
/// See [`handle_chat`] — same error contract.
async fn non_stream_chat_completion(
    state: ApiState,
    req: ChatRequest,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let response = handle_chat(&state, req).await?;
    Ok(Json(response).into_response())
}

/// Build the standard `engine_unavailable` `(StatusCode, Json<ErrorResponse>)`
/// pair returned when the engine channel is closed at request time.
fn engine_unavailable_error() -> (axum::http::StatusCode, Json<ErrorResponse>) {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse::with_code(
            "Engine unavailable",
            "server_error",
            "engine_unavailable",
        )),
    )
}

/// REL-01: build the standard `engine_overloaded` error returned when
/// the bounded engine mailbox is saturated. Clients should treat
/// this as retryable with backoff (the message explicitly suggests
/// `Retry-After`-style behavior).
fn engine_overloaded_error() -> (axum::http::StatusCode, Json<ErrorResponse>) {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse::with_code(
            "Engine overloaded; retry with backoff",
            "server_error",
            "engine_overloaded",
        )),
    )
}

// Unit tests are extracted to `tests.rs` to keep this handler file
// under the 800-line soft cap. The sibling covers the request
// validation + prompt-rendering paths; handler-level integration
// tests live under `tests/`.
#[cfg(test)]
mod tests;
