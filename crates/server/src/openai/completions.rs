//! `OpenAI` legacy Completions endpoint: `POST /v1/completions`. Prompt-string in, completion string out.
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

use super::sampling_validation::validate_sampling_params;
use super::types::{CompletionChoice, CompletionRequest, CompletionResponse, ErrorResponse, Usage};
use crate::ApiState;

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
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
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

    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
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

    // REL-01: use `try_send` so a saturated mailbox fails fast with
    // 503 `engine_overloaded` instead of blocking.
    state
        .engine_tx
        .try_send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
            seq_id_tx,
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
        let stream = stream::unfold(
            (response_rx, cancel_guard.clone()),
            move |(mut rx, cancel_guard)| {
                let tokenizer = tokenizer.clone();
                async move {
                    match rx.recv().await {
                        Some(token) => {
                            let text = tokenizer.decode(&[token]);
                            if should_skip_token_text(&tokenizer, &text) {
                                return Some((
                                    Ok::<Event, Infallible>(Event::default().data("")),
                                    (rx, cancel_guard),
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
                            Some((Ok(Event::default().data(sse_payload)), (rx, cancel_guard)))
                        }
                        None => {
                            // Natural completion — disarm so Drop
                            // doesn't send a redundant CancelRequest.
                            cancel_guard.disarm();
                            Some((Ok(Event::default().data("[DONE]")), (rx, cancel_guard)))
                        }
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

    let text = clean_completion_text(&state.tokenizer, &state.tokenizer.decode(&tokens));
    let choice = CompletionChoice {
        text,
        index: 0,
        finish_reason: Some("stop".to_string()),
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
