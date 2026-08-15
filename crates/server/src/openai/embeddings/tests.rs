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

// RIL ISS-068 / TASK-081: the sibling sync completions endpoint rejects an
// empty prompt (`req.prompt.is_empty()` -> 400 "prompt is required",
// completions.rs:1402). Embeddings accepted `input: [""]` / `["   "]` — the
// guard only checked the *array* was non-empty, so an empty/whitespace
// element flowed through to a zero-token embed the sibling contract forbids.

#[tokio::test]
async fn test_embeddings_rejects_empty_string_element() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec![String::new()],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_rejects_whitespace_only_element() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec!["   ".to_string()],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_rejects_any_empty_element_in_list() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec![
            "valid one".to_string(),
            String::new(),
            "valid two".to_string(),
        ],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_rejects_whitespace_element_in_list() {
    let state = create_test_state();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec![
            "valid one".to_string(),
            "\t\n".to_string(),
            "valid two".to_string(),
        ],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

// RIL ISS-068 / TASK-081: the request must bound the number of input
// elements. Each element produces a full model-dimension vector held in
// memory + serialized; with no cap (mirroring the `n <= 8` / `best_of <= 20`
// fan-out bounds on the generation siblings) a single request can amplify
// 1 MiB of tiny strings into an unbounded response.

#[tokio::test]
async fn test_embeddings_rejects_over_max_inputs() {
    let state = create_test_state();
    let too_many = (0..=MAX_EMBEDDINGS_INPUTS)
        .map(|_| "x".to_string())
        .collect();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: too_many,
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, _) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_embeddings_accepts_up_to_max_inputs() {
    let state = create_test_state();
    let inputs: Vec<String> = (0..MAX_EMBEDDINGS_INPUTS)
        .map(|_| "x".to_string())
        .collect();
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: inputs,
    };

    // Exactly MAX_EMBEDDINGS_INPUTS passes the per-request cap; the closed
    // test engine then surfaces as 503 (the fixture channel), proving the
    // gate did not reject the request.
    let result = embeddings(State(state), Json(req)).await;
    let (status, body) = result.unwrap_err();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body.error.code.as_deref(), Some("engine_unavailable"));
}

// RIL ISS-068 / TASK-081 (ISS-044 class): an input whose token count
// exceeds the model context must be rejected instead of forcing a
// full-length embed forward. Mirrors the chat/completions context-length
// gate; the tokenizer result for a long string is checked against the
// model's max length.

#[tokio::test]
async fn test_embeddings_rejects_input_exceeding_context_length() {
    let mut state = create_test_state();
    // Qwen3 no explicit max, so drive the gate via max_model_len. The
    // Tokenizer fixture is a stub that tokenises naively (1 token per
    // ASCII byte-ish), so use a long string vs a tiny max_model_len.
    state.max_model_len = Some(8);
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        input: vec!["this is a very long input string that certainly exceeds eight".to_string()],
    };

    let result = embeddings(State(state), Json(req)).await;
    assert!(result.is_err());
    let (status, body) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body.error.code.as_deref(), Some("context_length_exceeded"));
}

#[tokio::test]
async fn test_embeddings_accepts_input_within_context_length() {
    let mut state = create_test_state();
    state.max_model_len = Some(8);
    let req = EmbeddingsRequest {
        model: "test-model".to_string(),
        // "ab" tokenises to 2 tokens, within the 8-token context.
        input: vec!["ab".to_string()],
    };

    // Passes the context gate; the closed engine channel then 503s.
    let result = embeddings(State(state), Json(req)).await;
    let (status, body) = result.unwrap_err();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body.error.code.as_deref(), Some("engine_unavailable"));
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
