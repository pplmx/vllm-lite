//! `OpenAI` error-contract test matrix.
//!
//! Locks in the v0.1 server's wire-level error behavior so we don't silently
//! regress it. Covers:
//!
//! * Per-handler validation failures → `400 BAD_REQUEST` with `code` field
//!   carrying the OpenAI-standard `invalid_request_error` category.
//! * Engine-channel-closed failures → `503 SERVICE_UNAVAILABLE` with
//!   `code = "engine_unavailable"` (machine-readable retry hint).
//! * `ErrorResponse::with_code` constructs correctly (round-trips through
//!   JSON, code preserved).
//! * `ErrorResponse::new` produces `code = None` (backward compat).
//!
//! Run with: `cargo nextest run -p vllm-server --test error_contract`.

use axum::http::StatusCode;
use vllm_server::openai::types::ErrorResponse;

/// Build a test [`vllm_server::ApiState`] with the given architecture.
///
/// Convenience wrapper so test bodies stay focused on what's being asserted
/// rather than the fixture plumbing. The fixture uses a closed `engine_tx`
/// channel, which is exactly what we need to exercise the
/// `engine_unavailable` error path.
fn create_test_state() -> vllm_server::ApiState {
    vllm_server::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
}

/// Embeddings-capable variant: sets `arch_capabilities` to a
/// production profile so the embeddings handler's capability
/// gate is skipped. Used by the embeddings-specific tests in
/// this file; the `None` path is covered by
/// `embeddings_capability.rs`.
fn create_embeddings_capable_state() -> vllm_server::ApiState {
    use vllm_model::arch::ArchCapabilities;
    let mut state = create_test_state();
    state.arch_capabilities = Some(ArchCapabilities::PRODUCTION);
    state
}

// ---------------------------------------------------------------------------
// ErrorResponse constructors
// ---------------------------------------------------------------------------

#[test]
fn error_response_new_has_no_code() {
    // Backward-compat: legacy `new(message, type)` keeps `code = None`.
    // OpenAI clients built against earlier versions still see only
    // `error.message` + `error.type`.
    let resp = ErrorResponse::new("something went wrong", "server_error");
    assert_eq!(resp.error.message, "something went wrong");
    assert_eq!(resp.error.error_type, "server_error");
    assert!(resp.error.code.is_none());
}

#[test]
fn error_response_with_code_round_trips_through_serde() {
    let resp = ErrorResponse::with_code(
        "Model not found: qwen3-99b",
        "invalid_request_error",
        "model_not_found",
    );
    assert_eq!(resp.error.message, "Model not found: qwen3-99b");
    assert_eq!(resp.error.error_type, "invalid_request_error");
    assert_eq!(resp.error.code.as_deref(), Some("model_not_found"));

    // Serde round-trip preserves the code field. This is the contract that
    // OpenAI-compatible clients depend on.
    let json = serde_json::to_string(&resp).expect("serializable");
    assert!(json.contains(r#""code":"model_not_found""#), "got {json}");

    let parsed: ErrorResponse = serde_json::from_str(&json).expect("deserializable");
    assert_eq!(parsed.error.code.as_deref(), Some("model_not_found"));
}

#[test]
fn error_response_json_field_order_matches_openai_spec() {
    // OpenAI's error schema is an object with `error: { message, type, code? }`.
    // We verify the parent wrapper shape explicitly so a future refactor can't
    // accidentally rename the wrapper.
    let resp = ErrorResponse::with_code("oops", "server_error", "internal_error");
    let val: serde_json::Value = serde_json::to_value(&resp).expect("serializable");
    assert!(val.get("error").is_some(), "missing 'error' wrapper");
    let err = val.get("error").unwrap();
    assert!(err.get("message").is_some());
    assert!(err.get("type").is_some());
    assert!(err.get("code").is_some());
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — error matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_rejects_empty_model_with_400_and_invalid_request_code() {
    use axum::{Extension, Json, extract::State};
    use vllm_server::openai::chat::chat_completions;
    use vllm_server::openai::types::{ChatMessage, ChatRequest};
    use vllm_server::security::correlation::CorrelationId;

    let state = create_test_state();
    let req = ChatRequest {
        model: String::new(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
        }],
        temperature: None,
        top_p: None,
        max_tokens: None,
        stream: None,
        n: None,
        stop: None,
        user: None,
    };

    let result = chat_completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    let (status, body) = result.expect_err("expected error for empty model");
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body.error.error_type, "invalid_request_error");
}

#[tokio::test]
async fn chat_returns_503_with_engine_unavailable_code_when_channel_closed() {
    use axum::{Extension, Json, extract::State};
    use vllm_server::openai::chat::chat_completions;
    use vllm_server::openai::types::{ChatMessage, ChatRequest};
    use vllm_server::security::correlation::CorrelationId;

    let state = create_test_state(); // engine_tx has no receiver
    let req = ChatRequest {
        model: "qwen3".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            name: None,
        }],
        temperature: None,
        top_p: None,
        max_tokens: Some(16),
        stream: None,
        n: None,
        stop: None,
        user: None,
    };

    let result = chat_completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    let (status, body) = result.expect_err("expected engine-channel error");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body.error.code.as_deref(),
        Some("engine_unavailable"),
        "closed engine channel must surface the engine_unavailable code"
    );
}

// ---------------------------------------------------------------------------
// /v1/embeddings — error matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embeddings_rejects_empty_model_with_400() {
    use axum::{Json, extract::State};
    use vllm_server::openai::embeddings::embeddings;
    use vllm_server::openai::types::EmbeddingsRequest;

    let state = create_embeddings_capable_state();
    let req = EmbeddingsRequest {
        model: String::new(),
        input: vec!["hi".into()],
    };

    let result = embeddings(State(state), Json(req)).await;
    let (status, _) = result.expect_err("expected 400");
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn embeddings_returns_503_with_engine_unavailable_code_when_channel_closed() {
    use axum::{Json, extract::State};
    use vllm_server::openai::embeddings::embeddings;
    use vllm_server::openai::types::EmbeddingsRequest;

    let state = create_embeddings_capable_state();
    let req = EmbeddingsRequest {
        model: "qwen3".into(),
        input: vec!["hi".into()],
    };

    let result = embeddings(State(state), Json(req)).await;
    let (status, body) = result.expect_err("expected engine-channel error");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body.error.code.as_deref(), Some("engine_unavailable"));
}

// ---------------------------------------------------------------------------
// /v1/completions — error matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_rejects_empty_prompt_with_400() {
    use axum::{Extension, Json, extract::State};
    use vllm_server::openai::completions::completions;
    use vllm_server::openai::types::CompletionRequest;
    use vllm_server::security::correlation::CorrelationId;

    let state = create_test_state();
    let req = CompletionRequest {
        model: None,
        prompt: String::new(),
        temperature: None,
        top_p: None,
        max_tokens: None,
        stream: None,
        n: None,
        stop: None,
        user: None,
    };

    let result = completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    let (status, _) = result.expect_err("expected 400");
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn completions_returns_503_with_engine_unavailable_code_when_channel_closed() {
    use axum::{Extension, Json, extract::State};
    use vllm_server::openai::completions::completions;
    use vllm_server::openai::types::CompletionRequest;
    use vllm_server::security::correlation::CorrelationId;

    let state = create_test_state();
    let req = CompletionRequest {
        model: None,
        prompt: "hello".into(),
        temperature: None,
        top_p: None,
        max_tokens: Some(8),
        stream: None,
        n: None,
        stop: None,
        user: None,
    };

    let result = completions(
        State(state),
        Extension(CorrelationId("test-correlation-id".into())),
        Json(req),
    )
    .await;
    let (status, body) = result.expect_err("expected engine-channel error");
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body.error.code.as_deref(), Some("engine_unavailable"));
}
