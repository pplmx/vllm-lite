//! HTTP-level chat handler tests with a mock inference engine.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::post,
};
use http_body_util::BodyExt;
use tower::ServiceExt;
use vllm_model::config::Architecture;
use vllm_server::ApiState;
use vllm_server::openai::chat::chat_completions;
use vllm_server::openai::chat_template::ChatTemplate;
use vllm_server::openai::types::ChatMessage;
use vllm_server::test_fixtures::{api_state_with_mock_engine, spawn_mock_engine};

fn chat_request_json(model: &str, stream: bool) -> String {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
        "max_tokens": 3
    })
    .to_string()
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

#[tokio::test]
async fn test_chat_completions_rejects_empty_model() {
    let state = vllm_server::test_fixtures::api_state(Architecture::Qwen3);
    let app = router(state);

    let body = serde_json::json!({
        "model": "",
        "messages": [{"role": "user", "content": "Hi"}]
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_non_streaming_with_mock_engine() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json("test-model", false)))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn test_chat_prompt_format_follows_architecture() {
    use vllm_server::openai::chat::build_prompt_from_messages;

    let qwen_prompt = build_prompt_from_messages(
        ChatTemplate::for_architecture(Architecture::Qwen3),
        &[ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
    );
    assert!(qwen_prompt.contains("<|im_start|>"));

    let llama_prompt = build_prompt_from_messages(
        ChatTemplate::for_architecture(Architecture::Llama),
        &[ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }],
    );
    assert!(llama_prompt.starts_with("<|begin_of_text|>"));
    assert_ne!(qwen_prompt, llama_prompt);
}

#[tokio::test]
async fn test_chat_completions_streaming_returns_sse() {
    let (engine_tx, _handle) = spawn_mock_engine(vec![7, 8]);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Llama,
        batch_manager: Arc::new(vllm_server::openai::batch::manager::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(vllm_server::HealthChecker::new(
            true, true,
        ))),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json("llama-test", true)))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.contains("text/event-stream"));
}

/// API-01 regression: pre-fix the streaming handler concatenated the
/// final JSON chunk and the `[DONE]` sentinel into a single SSE event
/// (`"{json}\n\n[DONE]"`), which strict OpenAI SDK / SSE clients do
/// not parse. Post-fix the final chunk and `[DONE]` are separate
/// `data:` events; see `docs/technical-due-diligence/architecture-performance.md`
/// §5.1.3 and the v31.0 P4 follow-up batch.
#[tokio::test]
async fn test_chat_streaming_done_is_separate_event() {
    let (engine_tx, _handle) = spawn_mock_engine(vec![7, 8]);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Llama,
        batch_manager: Arc::new(vllm_server::openai::batch::manager::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(vllm_server::HealthChecker::new(
            true, true,
        ))),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json("llama-test", true)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    // Split on the SSE event terminator `\n\n` so we can count events.
    let events: Vec<&str> = body_str.split("\n\n").filter(|s| !s.is_empty()).collect();
    assert!(
        !events.is_empty(),
        "SSE stream should contain at least one event, body was: {body_str}"
    );

    // The very last event MUST be the `[DONE]` sentinel, and it MUST
    // NOT contain any JSON payload — strict clients parse each `data:`
    // field separately and reject `[DONE]` that carries JSON.
    let last = events.last().unwrap();
    assert!(
        last.contains("[DONE]"),
        "last SSE event should contain [DONE], got: {last}"
    );
    assert!(
        !last.contains("\"finish_reason\""),
        "[DONE] event must not contain JSON payload (pre-fix bug), got: {last}"
    );

    // The penultimate event(s) must contain the final chunk's JSON
    // payload — look for the finish_reason field, which the pre-fix
    // version never emitted on the streaming path.
    let final_chunk = events
        .iter()
        .rev()
        .find(|e| e.contains("\"finish_reason\""))
        .unwrap_or_else(|| panic!("no SSE event carried finish_reason; events: {events:?}"));
    assert!(
        final_chunk.contains("\"finish_reason\":\"stop\"")
            || final_chunk.contains("\"finish_reason\":\"length\""),
        "final chunk must carry a non-null finish_reason, got: {final_chunk}"
    );
}

/// API-01 regression: pre-fix the non-streaming chat handler
/// hardcoded `finish_reason: "stop"` even when the engine actually
/// stopped because the sequence hit `max_tokens`. Post-fix the
/// engine-supplied [`vllm_traits::FinishReason`] is mapped to the
/// OpenAI string (`"length"`).
#[tokio::test]
async fn test_chat_non_streaming_finish_reason_propagation() {
    // We use the default mock engine which does NOT send a finish
    // reason; the handler must fall back to `"stop"` rather than
    // panic or hang. The exact string is asserted here.
    let (engine_tx, _handle) = spawn_mock_engine(vec![7, 8, 9]);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Llama,
        batch_manager: Arc::new(vllm_server::openai::batch::manager::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(vllm_server::HealthChecker::new(
            true, true,
        ))),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json("llama-test", false)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    let finish_reason = body["choices"][0]["finish_reason"].as_str();
    assert_eq!(
        finish_reason,
        Some("stop"),
        "non-streaming mock should yield finish_reason=stop (mock drops the reason oneshot)"
    );
}

/// API-01 (technical due diligence §5.1): `n > 1` is declared in
/// `ChatRequest` but the engine emits exactly one completion per
/// request. Silent acceptance + ignored field would be a contract
/// violation — we return 400 invalid_request_error instead.
#[tokio::test]
async fn test_chat_rejects_n_greater_than_one_with_400() {
    let state = vllm_server::test_fixtures::api_state(Architecture::Qwen3);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 2,
        "max_tokens": 3,
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "n > 1 must be rejected at the HTTP boundary"
    );
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("n > 1"),
        "error message must name the rejected field, got: {}",
        body["error"]["message"]
    );
}

/// API-01 (technical due diligence §5.1): `n = 1` is the OpenAI
/// default and must NOT be rejected — it is functionally identical
/// to omitting the field.
#[tokio::test]
async fn test_chat_accepts_n_equal_to_one() {
    let (state, _handle) = api_state_with_mock_engine(Architecture::Qwen3, vec![10]);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 1,
        "max_tokens": 3,
    })
    .to_string();

    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "n = 1 must be accepted (equivalent to omitting the field)"
    );
}

/// API-01: non-empty `stop` is declared in `ChatRequest` but the
/// engine stops at `max_tokens` or natural EOS only. Accepting it
/// and ignoring it would silently truncate at `max_tokens` even
/// when a stop sequence was emitted.
#[tokio::test]
async fn test_chat_rejects_non_empty_stop_with_400() {
    let state = vllm_server::test_fixtures::api_state(Architecture::Qwen3);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["\n", "END"],
        "max_tokens": 3,
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "non-empty stop must be rejected at the HTTP boundary"
    );
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("stop sequences"),
        "error message must name the rejected field"
    );
}

/// API-01: empty `stop` array is functionally a no-op and must
/// pass through unchanged.
#[tokio::test]
async fn test_chat_accepts_empty_stop_array() {
    let (state, _handle) = api_state_with_mock_engine(Architecture::Qwen3, vec![10]);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [],
        "max_tokens": 3,
    })
    .to_string();

    let response = router(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "empty stop array must be accepted (no-op)"
    );
}
