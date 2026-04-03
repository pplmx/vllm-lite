#![allow(dead_code)]

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

use super::types::*;
use crate::ApiState;

#[allow(dead_code)]
pub(crate) async fn completions(
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
    let max_tokens = req.max_tokens.unwrap_or(100) as usize;
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
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new("Engine unavailable", "internal_error")),
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
                        if text.is_empty() {
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
                        let data = chunk.to_string();
                        Some((Ok(Event::default().data(data)), rx))
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

    let text = state.tokenizer.decode(&tokens);
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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[tokio::test]
    async fn test_completions_empty_prompt() {
        // Test with empty prompt - should return 400
    }
}
