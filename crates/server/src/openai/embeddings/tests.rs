//! Unit tests for the `OpenAI` `/v1/embeddings` endpoint.
//!
//! Covers the validation gates and the engine-channel error mapping:
//!
//! - empty `model` → 400 `BAD_REQUEST`
//! - empty `input` → 400 `BAD_REQUEST`
//! - non-empty payload with closed engine channel → 503
//!   `SERVICE_UNAVAILABLE` with `engine_unavailable` error code
//!   (clients retry on this since it's a transient server-side issue).
//!
//! All tests rely on `crate::test_fixtures::api_state(Qwen3)` to
//! stand up an `ApiState` without a live engine; the closed
//! `engine_tx` channel is what surfaces as 503.
//!
//! Production-readiness §10: the capability gate runs first —
//! we set `arch_capabilities = Some(PRODUCTION)` so the
//! validation paths are exercised. The `None` path
//! (`embeddings_unsupported`) is covered by the
//! `embeddings_capability.rs` integration test.
use super::*;

use axum::http::StatusCode;
use vllm_model::arch::ArchCapabilities;

fn create_test_state() -> crate::ApiState {
    let mut state = crate::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3);
    // Skip the embeddings-capability gate so the per-field
    // validation paths (empty model / empty input / closed
    // engine) are exercised.
    state.arch_capabilities = Some(ArchCapabilities::PRODUCTION);
    state
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
