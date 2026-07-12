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
