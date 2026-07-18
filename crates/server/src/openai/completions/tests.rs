//! Unit tests for the `OpenAI` `/v1/completions` endpoint.
//!
//! Covers the validation path (empty prompt → 400) and the
//! engine-channel error mapping (closed channel → 503
//! `engine_unavailable`). The handler is exercised without a live
//! engine by relying on `test_fixtures::api_state`.
use super::*;
use crate::security::correlation::CorrelationId;
use axum::Extension;

fn create_test_state() -> crate::ApiState {
    crate::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
}

#[tokio::test]
async fn test_completions_empty_prompt() {
    let state = create_test_state();
    let req = CompletionRequest {
        model: None,
        prompt: String::new(),
        temperature: None,
        top_p: None,
        max_tokens: Some(100),
        stream: None,
        n: None,
        stop: None,
        user: None,
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
    };

    let result = completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
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
        top_p: None,
        max_tokens: Some(10),
        stream: None,
        n: None,
        stop: None,
        user: None,
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
    };

    // With no engine running, this will fail to send to engine
    // but we can verify it doesn't fail on validation. The closed-channel
    // error surfaces as 503 SERVICE_UNAVAILABLE with `engine_unavailable`
    // code (see `completions` handler) — distinguishable from a real
    // server-side bug, and safe for clients to retry.
    let result = completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    assert!(result.is_err());
    let (status, body) = result.unwrap_err();
    assert_eq!(status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body.error.code.as_deref(), Some("engine_unavailable"));
}
