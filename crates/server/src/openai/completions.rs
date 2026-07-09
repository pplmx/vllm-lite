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

    let (response_tx, mut response_rx) = mpsc::channel(64);

    state
        .engine_tx
        .send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .map_err(|_| {
            (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::with_code(
                    "Engine unavailable",
                    "server_error",
                    "engine_unavailable",
                )),
            )
        })?;

    if is_streaming {
        let tokenizer = state.tokenizer.clone();
        let stream = stream::unfold(response_rx, move |mut rx| {
            let tokenizer = tokenizer.clone();
            async move {
                match rx.recv().await {
                    Some(token) => {
                        let text = tokenizer.decode(&[token]);
                        if should_skip_token_text(&tokenizer, &text) {
                            return Some((Ok::<Event, Infallible>(Event::default().data("")), rx));
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
                        Some((Ok(Event::default().data(sse_payload)), rx))
                    }
                    None => Some((Ok(Event::default().data("[DONE]")), rx)),
                }
            }
        });

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

// Unit tests live in `tests.rs` (sibling) to keep this handler file
// under the 800-line soft cap. They cover the empty-prompt validation
// path and the engine-channel error mapping (closed channel → 503
// `engine_unavailable`).
#[cfg(test)]
mod tests;
