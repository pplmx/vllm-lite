//! `OpenAI` legacy Completions endpoint: `POST /v1/completions`. Prompt-string in, completion string out.
use axum::{
    Extension, Json,
    extract::State,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
};
use futures::stream;
use std::convert::Infallible;
use tokio::sync::mpsc;

use super::sampling_validation::{validate_completion_request_fields, validate_sampling_params};
use super::types::{CompletionChoice, CompletionRequest, CompletionResponse, ErrorResponse, Usage};
use crate::ApiState;
use crate::security::correlation::CorrelationId;

fn should_skip_token_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> bool {
    text.is_empty() || tokenizer.is_special_token(text)
}

fn clean_completion_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> String {
    tokenizer.clean_special_tokens(text)
}

/// OpenAI-compatible `/v1/completions` HTTP handler. Dispatches to streaming
/// (SSE) or non-streaming based on `req.stream`.
///
/// Validates that `prompt` is non-empty and forwards an
/// [`vllm_core::types::EngineMessage::AddRequest`] to the engine for each call.
///
/// # Errors
///
/// Returns `(StatusCode, ErrorResponse)` when:
/// - prompt is empty (`BAD_REQUEST`)
/// - the engine channel is closed (`SERVICE_UNAVAILABLE`, code `engine_unavailable`)
/// - token decoding or SSE serialization fails
///
/// # Panics
///
/// Panics if the streaming path reaches a `seq_id_rx.expect(...)` after
/// the `is_streaming` branch already established that the oneshot was
/// constructed. This invariant is held by the surrounding `if
/// is_streaming { ... }` guard; the panic exists as a tripwire for
/// future refactors that might break the link between the two branches.
pub async fn completions(
    State(state): State<ApiState>,
    Extension(correlation_id): Extension<CorrelationId>,
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    // API-01: reject OpenAI fields the engine does not yet honour
    // BEFORE doing any work. Mirror of chat.rs:
    // `validate_chat_request_fields`. Honest 400 > silent degradation.
    validate_completion_request_fields(&req)?;

    if req.prompt.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "prompt is required",
                "invalid_request_error",
            )),
        ));
    }

    let is_streaming = req.stream.unwrap_or(false);
    let prompt = req.prompt;
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens_len = prompt_tokens.len();
    let max_tokens = usize::try_from(req.max_tokens.unwrap_or(100)).unwrap_or(100);
    let total_max = prompt_tokens_len + max_tokens;

    // Production-readiness §4: reject requests whose
    // prompt + max_tokens would exceed the model's context
    // length. See chat.rs for the full rationale.
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

    // Forward `top_p` to the engine (mirror of the chat handler).
    // The engine's `sample_batch_with_params` honours `top_p` via
    // nucleus sampling; the value is range-checked by
    // `validate_completion_request_fields` earlier in this handler.
    if let Some(top_p) = req.top_p {
        request.sampling_params.top_p = top_p;
    }

    // Forward `frequency_penalty` to the engine's existing
    // `repeat_penalty` slot (P27 v0.3 wire-type follow-up; P29
    // closes the boost-semantics carve-out via the sign-aware
    // engine refactor). Mirrors the chat handler's wire-through
    // so the legacy `/v1/completions` endpoint sees the same
    // penalty behavior (including the boost semantic for negative
    // values). See the chat handler's matching block for the full
    // rationale on `(1.0 + fp).max(1e-3)`.
    if let Some(fp) = req.frequency_penalty {
        request.sampling_params.repeat_penalty = (1.0 + fp).max(1e-3);
    }

    // Forward `presence_penalty` to the engine's
    // `SamplingParams::presence_penalty` slot (P28 v0.3
    // wire-type follow-up — engine wire-through). Mirrors the
    // chat handler so the legacy endpoint sees the same penalty
    // behavior. Unlike `frequency_penalty` (clamped via `max(1.0,
    // ...)` because of the logit-divide sign-flip bug for negative
    // values), `presence_penalty` is an additive bias so the value
    // is forwarded verbatim — no clamping needed.
    if let Some(pp) = req.presence_penalty {
        request.sampling_params.presence_penalty = pp;
    }

    // Forward `logit_bias` to the engine's new
    // `SamplingParams::logit_bias` slot (P30 v0.3 wire-type
    // follow-up — engine wire-through). Mirrors the chat handler
    // so the legacy endpoint sees the same bias semantics. The
    // engine's `apply_logit_bias` adds each map value to the logit
    // at the corresponding token position before the temperature /
    // top-k / top-p pipeline. Per OpenAI spec the values are
    // constrained to the `[-100, 100]` range; the validator
    // (`validate_completion_request_fields`) rejects NaN /
    // ±infinity / out-of-range values with `400`, so we only need
    // to forward the field here. The completions handler does not
    // currently log the field (parity with the `seed` / `user` /
    // `frequency_penalty` / `presence_penalty` fields — chat
    // handler logs them, completions handler does not).
    if let Some(ref lb) = req.logit_bias {
        request.sampling_params.logit_bias = Some(lb.clone());
    }

    // Forward `seed` to the engine's `SamplingParams::seed` slot
    // (P34 v0.2 wire-type follow-up — engine wire-through). Same
    // `i64 as u64` cast rationale as the chat handler — see that
    // file for the full discussion. The legacy completions handler
    // does not currently log the seed field (parity with `user` /
    // `frequency_penalty` / `presence_penalty` / `logit_bias` —
    // chat handler logs them, completions handler accepts them at
    // the wire type but does not log).
    if let Some(seed) = req.seed {
        request.sampling_params.seed = Some(seed as u64);
    }

    // Reject sampling parameters the engine cannot honour (currently
    // beam_width > 1) BEFORE enqueuing — see `sampling_validation`.
    validate_sampling_params(&request.sampling_params)?;

    // Production-readiness recommendation: streaming variants
    // allocate a seq_id round-trip oneshot so we can learn the
    // engine-assigned id and forward `EngineMessage::CancelRequest`
    // if the client disconnects mid-stream. Without this, the
    // engine keeps generating tokens for a caller that has
    // already gone away.
    let (seq_id_tx, seq_id_rx) = if is_streaming {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let (response_tx, mut response_rx) = mpsc::channel(64);
    // API-01 finish_reason propagation (mirrors chat.rs): the engine
    // sends the [`FinishReason`] through this oneshot just before it
    // drops the token response channel, so we can emit the
    // OpenAI-correct `finish_reason` (`"length"` when the sequence hit
    // `max_tokens`, instead of the pre-fix hardcoded `"stop"`).
    let (finish_reason_tx, finish_reason_rx) = tokio::sync::oneshot::channel();

    // REL-01: use `try_send` so a saturated mailbox fails fast with
    // 503 `engine_overloaded` instead of blocking.
    state
        .engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
            seq_id_tx,
            finish_reason_tx: Some(finish_reason_tx),
            // Production-readiness §6: forward the correlation id
            // (same rationale as the chat handler). The engine run
            // loop's `tracing::info_span!("engine.add_request",
            // request_id)` attaches it to every synchronous log
            // line in add_request and its callees.
            request_id: Some(correlation_id.0.clone()),
        })
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => overload_response(),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => unavailable_response(),
        })?;

    if is_streaming {
        // Block briefly until the engine assigns the seq_id (see
        // chat.rs for rationale on the 1 s cap).
        let seq_id: vllm_traits::SeqId = match tokio::time::timeout(
            std::time::Duration::from_secs(1),
            seq_id_rx.expect("seq_id_rx is set when is_streaming"),
        )
        .await
        {
            Ok(Ok(id)) => id,
            _ => return Err(unavailable_response()),
        };

        // Drop guard — see `chat::CancelOnDrop` for full rationale.
        let cancel_guard = std::sync::Arc::new(crate::openai::chat::CancelOnDrop {
            engine_tx: state.engine_tx.clone(),
            seq_id: std::sync::atomic::AtomicU64::new(seq_id),
            fired: std::sync::atomic::AtomicBool::new(false),
            request_id: format!(
                "cmpl_{}",
                uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
            ),
        });

        let tokenizer = state.tokenizer.clone();
        // API-01 finish_reason propagation + `[DONE]` split: mirror
        // `chat.rs` — final chunk carries the real `finish_reason`
        // and `[DONE]` is a separate SSE event. See
        // `docs/technical-due-diligence/architecture-performance.md` §5.1.3.
        enum Terminal {
            Streaming,
            EmitDoneSentinel,
            Done,
        }
        let stream = stream::unfold(
            (
                response_rx,
                cancel_guard.clone(),
                Some(finish_reason_rx),
                Terminal::Streaming,
            ),
            move |(mut rx, cancel_guard, mut reason_rx_opt, mut terminal)| {
                let tokenizer = tokenizer.clone();
                async move {
                    match terminal {
                        Terminal::Done => None,
                        Terminal::EmitDoneSentinel => {
                            terminal = Terminal::Done;
                            Some((
                                Ok::<Event, Infallible>(Event::default().data("[DONE]")),
                                (rx, cancel_guard, reason_rx_opt, terminal),
                            ))
                        }
                        Terminal::Streaming => match rx.recv().await {
                            Some(token) => {
                                let text = tokenizer.decode(&[token]);
                                if should_skip_token_text(&tokenizer, &text) {
                                    return Some((
                                        Ok::<Event, Infallible>(Event::default().data("")),
                                        (rx, cancel_guard, reason_rx_opt, terminal),
                                    ));
                                }
                                let chunk = serde_json::json!({
                                    "id": "cmpl-stream",
                                    "object": "text_completion",
                                    "choices": [{
                                        "text": text,
                                        "index": 0,
                                    }]
                                });
                                let sse_payload = chunk.to_string();
                                Some((
                                    Ok(Event::default().data(sse_payload)),
                                    (rx, cancel_guard, reason_rx_opt, terminal),
                                ))
                            }
                            None => {
                                let reason_string = if let Some(rx) = reason_rx_opt.take() {
                                    match rx.await {
                                        Ok(vllm_traits::FinishReason::Length) => "length",
                                        Ok(vllm_traits::FinishReason::Stop) => "stop",
                                        Ok(vllm_traits::FinishReason::Cancelled) => "stop",
                                        Err(_) => "stop",
                                    }
                                } else {
                                    "stop"
                                };
                                // Natural completion — disarm so Drop
                                // doesn't send a redundant CancelRequest.
                                cancel_guard.disarm();
                                let chunk = serde_json::json!({
                                    "id": "cmpl-stream",
                                    "object": "text_completion",
                                    "choices": [{
                                        "text": "",
                                        "index": 0,
                                        "finish_reason": reason_string,
                                    }]
                                });
                                let sse_payload = chunk.to_string();
                                terminal = Terminal::EmitDoneSentinel;
                                Some((
                                    Ok(Event::default().data(sse_payload)),
                                    (rx, cancel_guard, reason_rx_opt, terminal),
                                ))
                            }
                        },
                    }
                }
            },
        );

        return Ok(Sse::new(Box::pin(stream)).into_response());
    }

    // 非流式 - 返回普通 JSON
    let mut tokens = Vec::new();
    while let Some(token) = response_rx.recv().await {
        tokens.push(token);
    }

    // Engine sends the reason before closing the response channel, so
    // this resolves immediately in the normal case. Fall back to
    // `"stop"` only when the oneshot was dropped without a value
    // (e.g. engine panicked between the two steps).
    let finish_reason = match finish_reason_rx.await {
        Ok(vllm_traits::FinishReason::Length) => "length".to_string(),
        Ok(vllm_traits::FinishReason::Stop) => "stop".to_string(),
        Ok(vllm_traits::FinishReason::Cancelled) => "stop".to_string(),
        Err(_) => "stop".to_string(),
    };

    let text = clean_completion_text(&state.tokenizer, &state.tokenizer.decode(&tokens));
    let choice = CompletionChoice {
        text,
        index: 0,
        finish_reason: Some(finish_reason),
    };

    let usage = Usage::new(prompt_tokens_len, tokens.len());
    let response = CompletionResponse::new(
        format!("cmpl-{}", uuid::Uuid::new_v4()),
        req.model.unwrap_or_else(|| "default".to_string()),
        vec![choice],
        usage,
    );

    Ok(Json(response).into_response())
}

/// REL-01: 503 response returned when the bounded engine mailbox is
/// saturated (`mpsc::error::TrySendError::Full`). Distinct from
/// `unavailable_response` so clients can implement smarter retry
/// (backoff + jitter) for transient overload.
fn overload_response() -> (axum::http::StatusCode, Json<ErrorResponse>) {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse::with_code(
            "Engine overloaded; retry with backoff",
            "server_error",
            "engine_overloaded",
        )),
    )
}

/// 503 response returned when the engine channel is closed
/// (`mpsc::error::TrySendError::Closed`). Distinct from
/// `overload_response` so clients know not to retry — the engine is
/// gone.
fn unavailable_response() -> (axum::http::StatusCode, Json<ErrorResponse>) {
    (
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse::with_code(
            "Engine unavailable",
            "server_error",
            "engine_unavailable",
        )),
    )
}

// Unit tests live in `tests.rs` (sibling) to keep this handler file
// under the 800-line soft cap. They cover the empty-prompt validation
// path and the engine-channel error mapping (closed channel → 503
// `engine_unavailable`).
#[cfg(test)]
mod tests;
