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

fn should_skip_token_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> bool {
    text.is_empty() || tokenizer.is_special_token(text)
}

fn clean_completion_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> String {
    tokenizer.clean_special_tokens(text)
}

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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::openai::batch::manager::BatchManager;
    use std::sync::Arc;
    use vllm_model::tokenizer::Tokenizer;

    fn create_test_state() -> crate::ApiState {
        use vllm_core::metrics::EnhancedMetricsCollector;
        let tokenizer = Tokenizer::new();
        let (engine_tx, _engine_rx) = mpsc::unbounded_channel();
        crate::ApiState {
            engine_tx,
            tokenizer: Arc::new(tokenizer),
            batch_manager: Arc::new(BatchManager::new()),
            auth: None,
            health: Arc::new(std::sync::RwLock::new(crate::HealthChecker::new(
                true, true,
            ))),
            metrics: Arc::new(EnhancedMetricsCollector::new()),
        }
    }

    #[tokio::test]
    async fn test_completions_empty_prompt() {
        let state = create_test_state();
        let req = CompletionRequest {
            model: None,
            prompt: "".to_string(),
            temperature: None,
            max_tokens: Some(100),
            stream: None,
            n: None,
            stop: None,
        };

        let result = completions(State(state), Json(req)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_completions_with_valid_max_tokens() {
        let state = create_test_state();
        let req = CompletionRequest {
            model: None,
            prompt: "Hello".to_string(),
            temperature: None,
            max_tokens: Some(10),
            stream: None,
            n: None,
            stop: None,
        };

        // With no engine running, this will fail to send to engine
        // but we can verify it doesn't fail on validation
        let result = completions(State(state), Json(req)).await;
        // Expected: fails because engine channel has no receiver
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    }
}
