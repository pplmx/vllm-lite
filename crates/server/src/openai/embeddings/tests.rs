//! Unit tests for the OpenAI `embeddings` endpoint.
//!
//! Covers the validation gates and the engine-channel error mapping:
//!
//! - empty `model` → 400 BAD_REQUEST
//! - empty `input` → 400 BAD_REQUEST
//! - non-empty payload with closed engine channel → 503
//!   SERVICE_UNAVAILABLE with `engine_unavailable` error code
//!   (clients retry on this since it's a transient server-side issue).
//!
//! All tests rely on `crate::test_fixtures::api_state(Qwen3)` to
//! stand up an `ApiState` without a live engine; the closed
//! `engine_tx` channel is what surfaces as 503.
use super::*;

use axum::http::StatusCode;

fn create_test_state() -> crate::ApiState {
    crate::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
}

#[tokio::test]
async fn test_embeddings_empty_model() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: String::new(),
        input: vec!["test input".to_string()],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_empty_input() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec![],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_multiple_inputs() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec!["input1".to_string(), "input2".to_string()],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    // The test fixture's `engine_tx` is a closed mpsc channel; the handler
    // surfaces that as a 503 SERVICE_UNAVAILABLE with `code = "engine_unavailable"`
    // so clients know the failure is transient and retryable.
    let (status, body) = result.unwrap_err();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body.error.code.as_deref(),
        Some("engine_unavailable"),
        "error code must be machine-readable"
    );
}
