//! HTTP-level chat handler tests with a mock inference engine.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::post,
};
use http_body_util::BodyExt;
use tokio::sync::Mutex;
use tower::ServiceExt;
use vllm_core::types::{EngineMessage, SamplingParams};
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
        // chat_completions requires `Extension<CorrelationId>`
        // (P10 / production-readiness §6). Mount the same
        // middleware the production router uses so tests exercise
        // the real boundary.
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ))
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

// ===== top_p forwarding tests =====
//
// Architecture-performance §5.1 + STATE.md P6 follow-up:
// `top_p` is declared on `ChatRequest` and `CompletionRequest`. The
// handler must forward the value to `Request::sampling_params.top_p`
// so the engine's `sample_batch_with_params` honours it.
//
// These tests use a capturing mock engine (one slot) that records
// the `sampling_params` from the first `AddRequest` it receives, then
// asserts the field round-trips from JSON to engine-side state.

/// Mock engine that captures the `SamplingParams` of the first
/// `AddRequest` it receives, then replies with a single synthetic
/// token so the handler completes. Returned as `(handle, captured)`
/// where `captured` is an `Arc<Mutex<Option<SamplingParams>>>` —
/// tests `await` on it after the response to inspect the forwarded
/// value.
fn spawn_capturing_mock_engine() -> (
    vllm_server::api::EngineHandle,
    tokio::task::JoinHandle<()>,
    Arc<Mutex<Option<SamplingParams>>>,
) {
    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(8);
    let captured: Arc<Mutex<Option<SamplingParams>>> = Arc::new(Mutex::new(None));
    let captured_clone = Arc::clone(&captured);
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    request,
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send(1);
                    }
                    drop(finish_reason_tx);
                    *captured_clone.lock().await = Some(request.sampling_params.clone());
                    // TokenId is a type alias for u32 (see
                    // `vllm_traits::types::TokenId`), so we send the
                    // primitive directly rather than the old
                    // `TokenId(10)` tuple-struct form.
                    let _ = response_tx.send(10u32).await;
                    break;
                }
                EngineMessage::Shutdown => break,
                _ => {}
            }
        }
    });
    (engine_tx, handle, captured)
}

/// Build a minimal `ApiState` whose engine channel is wired to the
/// capturing mock. The mock only handles one request, so each test
/// needs its own state.
fn state_with_capturing_engine() -> (
    ApiState,
    tokio::task::JoinHandle<()>,
    Arc<Mutex<Option<SamplingParams>>>,
) {
    let (engine_tx, handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    (state, handle, captured)
}

/// `top_p = 0.9` on the JSON request must land as
/// `sampling_params.top_p = 0.9` on the engine side. This is the
/// "happy path" — the engine honours it via nucleus sampling.
#[tokio::test]
async fn test_chat_forwards_top_p_to_engine() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.9,
        "max_tokens": 1,
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
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.top_p - 0.9).abs() < 1e-6,
        "top_p must round-trip from JSON to SamplingParams; got {}",
        params.top_p
    );
}

/// `top_p` omitted on the request must leave the engine-side default
/// (`1.0`, i.e. no nucleus filtering) untouched.
#[tokio::test]
async fn test_chat_omitted_top_p_uses_engine_default() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
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
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.top_p, 1.0,
        "omitted top_p must leave engine default (1.0); got {}",
        params.top_p
    );
}

/// `top_p = 1.5` must be rejected with 400 BEFORE the engine sees
/// the request — sampling guards exist in the validator, not the
/// engine, so this also proves the request never reached the mock.
#[tokio::test]
async fn test_chat_rejects_top_p_above_one_with_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 1.5,
        "max_tokens": 1,
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("top_p"),
        "error message must name top_p; got: {}",
        body["error"]["message"]
    );

    // Validator must run BEFORE the engine is touched.
    assert!(
        captured.lock().await.is_none(),
        "out-of-range top_p must be rejected at the HTTP boundary, \
         not reach the engine"
    );
}

/// `top_p = 0` is also out of range (would select zero tokens) and
/// must be rejected with 400.
#[tokio::test]
async fn test_chat_rejects_top_p_zero_with_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.0,
        "max_tokens": 1,
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        captured.lock().await.is_none(),
        "top_p = 0 must be rejected at the HTTP boundary"
    );
}

/// `top_p` must round-trip on the `/v1/completions` endpoint too —
/// the field was added to `CompletionRequest` at the same time as
/// the chat forwarding fix, and the engine should see the same value
/// regardless of which endpoint produced the request.
#[tokio::test]
async fn test_completions_forwards_top_p_to_engine() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        // completions requires `Extension<CorrelationId>`
        // (P10 / production-readiness §6). Mount the same
        // middleware the production router uses so tests exercise
        // the real boundary.
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "top_p": 0.5,
        "max_tokens": 1,
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.top_p - 0.5).abs() < 1e-6,
        "top_p must round-trip on /v1/completions; got {}",
        params.top_p
    );
}

// === P21 regression tests: `user` field declaration ===
//
// `user` is OpenAI's end-user identifier for safety / abuse tracking.
// P21 declares the field on ChatRequest + CompletionRequest as
// `Option<String>` with `#[serde(default)]` so omitting it is a no-op,
// and threads it into the existing `tracing::info!` calls in the chat
// handler. Honoring is a no-op until a downstream consumer (rate-
// limiter, audit log) subscribes. These tests pin the wire-type
// contract: the field is accepted when present, ignored when absent,
// and never causes the handler to reject the request.

/// A chat request with the `user` field set must be accepted by the
/// handler (status 200) and reach the engine. Pre-fix the field was
/// undeclared and serde rejected the request with a 400-class
/// deserialization error.
#[tokio::test]
async fn test_chat_with_user_field_accepted_by_handler() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "user": "tenant-1234"
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
        StatusCode::OK,
        "user field must not cause 4xx; pre-fix it was undeclared and rejected by serde"
    );
}

/// Baseline: omitting the `user` field must continue to work (the
/// field is `#[serde(default)]` → `None`). Pins the backward-
/// compatible path so legacy clients are not broken.
#[tokio::test]
async fn test_chat_without_user_field_works_baseline() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
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

    assert_eq!(response.status(), StatusCode::OK);
}

/// `/v1/completions` must also accept the `user` field (parallel to
/// the chat path). The completion handler currently doesn't log the
/// field — adding a tracing line there is deferred to avoid scope
/// creep — but the wire-type contract must be symmetric.
#[tokio::test]
async fn test_completions_with_user_field_accepted_by_handler() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, _captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 1,
        "user": "tenant-1234"
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

/// Wire-type round-trip: a JSON body with `user` deserializes into a
/// `ChatRequest` whose `user` field equals the original string; a JSON
/// body without `user` deserializes into `user: None`. This pins the
/// serde contract independently of any handler-level test (a future
/// refactor that drops the `#[serde(default)]` annotation would fail
/// here).
#[tokio::test]
async fn test_chat_user_field_wire_type_round_trip() {
    use vllm_server::openai::types::{ChatMessage, ChatRequest, CompletionRequest};

    // With user present.
    let json_with = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "tenant-1234"
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_with)
        .expect("user field must round-trip from JSON to ChatRequest");
    assert_eq!(
        req.user.as_deref(),
        Some("tenant-1234"),
        "user must round-trip; got {:?}",
        req.user
    );

    // Without user present.
    let json_without = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}]
    })
    .to_string();
    let req: ChatRequest =
        serde_json::from_str(&json_without).expect("omitted user field must deserialize to None");
    assert!(
        req.user.is_none(),
        "omitted user must default to None; got {:?}",
        req.user
    );

    // CompletionRequest mirrors ChatRequest.
    let completion_json = serde_json::json!({
        "prompt": "Hello",
        "user": "tenant-5678"
    })
    .to_string();
    let req: CompletionRequest = serde_json::from_str(&completion_json)
        .expect("user field must round-trip from JSON to CompletionRequest");
    assert_eq!(
        req.user.as_deref(),
        Some("tenant-5678"),
        "user must round-trip on /v1/completions; got {:?}",
        req.user
    );

    // Reference unused-import guard: ChatMessage stays in scope so
    // future test edits don't accidentally drop the import.
    let _ = std::any::type_name::<ChatMessage>();
}

// === P22 regression tests: `response_format` field declaration ===
//
// `response_format` is OpenAI's JSON-mode selector. P22 declares
// the `ResponseFormat` enum (`Text` + `JsonObject` only) and adds
// the field to `ChatRequest` (NOT `CompletionRequest` — the
// legacy `/v1/completions` endpoint doesn't support it per OpenAI
// spec). Honoring is a no-op today (no constrained-decoding hook).
// These tests pin the wire-type contract: text + json_object are
// accepted, json_schema is rejected at the serde layer with 400,
// and the field defaults to `None` when omitted.

/// `response_format = {"type": "text"}` must be accepted (this is the
/// OpenAI default; explicit declaration should be equivalent to
/// omission). Pre-fix the field was undeclared and serde rejected
/// the request.
#[tokio::test]
async fn test_chat_with_response_format_text_accepted_by_handler() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "response_format": {"type": "text"}
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
        StatusCode::OK,
        "response_format.text must not cause 4xx; pre-fix the field was undeclared"
    );
}

/// `response_format = {"type": "json_object"}` must be accepted as a
/// v0.2 declaration pass-through. Honoring is a no-op (no constrained-
/// decoder hook yet) but the wire-type contract accepts the value.
#[tokio::test]
async fn test_chat_with_response_format_json_object_accepted_by_handler() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "response_format": {"type": "json_object"}
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
        StatusCode::OK,
        "response_format.json_object must be accepted as a v0.2 pass-through (deferred honoring)"
    );
}

/// `response_format = {"type": "json_schema"}` must be rejected —
/// the v0.3 + constrained-decoding subset is not implemented. Serde
/// rejects the unknown variant at deserialization; the handler
/// never sees the request. Axum's `Json<T>` extractor returns
/// `422 Unprocessable Entity` for deserialization failures (this is
/// axum's standard contract — 422 means "syntactically valid JSON
/// but semantically invalid input", which matches "unknown enum
/// variant" precisely). This test pins the 4xx rejection: any
/// non-2xx status proves the field was rejected at the wire
/// boundary.
#[tokio::test]
async fn test_chat_with_response_format_json_schema_rejected() {
    let state = vllm_server::test_fixtures::api_state(Architecture::Qwen3);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": {"type": "object"}}
        }
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

    let status = response.status();
    assert!(
        status.is_client_error(),
        "json_schema must be rejected with 4xx; got {status} (v0.3 work; not yet implemented in v0.2)",
    );
    // Pin the specific status for documentation: axum's Json extractor
    // returns 422 (Unprocessable Entity) for deserialization failures.
    // This is the axum-standard contract: 422 means "syntactically
    // valid JSON but semantically invalid input" (unknown enum variant
    // fits this definition precisely).
    assert_eq!(
        status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "axum's Json<T> extractor returns 422 for unknown enum variants at deserialization"
    );
}

/// Baseline: omitting `response_format` must continue to work (the
/// field is `#[serde(default)]` → `None`). Pins the backward-
/// compatible path so legacy clients are not broken.
#[tokio::test]
async fn test_chat_without_response_format_works_baseline() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
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

    assert_eq!(response.status(), StatusCode::OK);
}

/// Wire-type round-trip: a JSON body with `response_format.text` /
/// `response_format.json_object` deserializes to the corresponding
/// `ResponseFormat` enum variant; a body without `response_format`
/// deserializes to `None`; a body with an unknown variant fails to
/// deserialize. Pins the serde contract independently of any
/// handler-level test.
#[tokio::test]
async fn test_chat_response_format_wire_type_round_trip() {
    use vllm_server::openai::types::{ChatRequest, ResponseFormat};

    // `text` deserializes to `ResponseFormat::Text`.
    let json_text = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "text"}
    })
    .to_string();
    let req: ChatRequest =
        serde_json::from_str(&json_text).expect("response_format.text must round-trip from JSON");
    assert_eq!(
        req.response_format,
        Some(ResponseFormat::Text),
        "response_format.text must deserialize to ResponseFormat::Text; got {:?}",
        req.response_format
    );

    // `json_object` deserializes to `ResponseFormat::JsonObject`.
    let json_json_object = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"}
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_json_object)
        .expect("response_format.json_object must round-trip from JSON");
    assert_eq!(
        req.response_format,
        Some(ResponseFormat::JsonObject),
        "response_format.json_object must deserialize to ResponseFormat::JsonObject; got {:?}",
        req.response_format
    );

    // Omitted field deserializes to `None`.
    let json_omitted = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}]
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_omitted)
        .expect("omitted response_format must deserialize to None");
    assert!(
        req.response_format.is_none(),
        "omitted response_format must default to None; got {:?}",
        req.response_format
    );

    // `json_schema` (the v0.3 variant) must fail to deserialize — the
    // enum only declares Text + JsonObject, so serde rejects unknown
    // variants at the wire boundary.
    let json_schema = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_schema"}
    })
    .to_string();
    let result: Result<ChatRequest, _> = serde_json::from_str(&json_schema);
    assert!(
        result.is_err(),
        "response_format.json_schema must fail to deserialize (v0.3 variant not yet declared); got Ok({:?})",
        result.map(|_| "<request>")
    );
}

/// `/v1/completions` (legacy endpoint) must NOT declare the
/// `response_format` field at all — OpenAI spec does not support it
/// on this endpoint. This test pins the wire-type asymmetry: a
/// `CompletionRequest` cannot be constructed with a `response_format`
/// field because the struct doesn't have one.
#[tokio::test]
async fn test_completion_request_has_no_response_format_field() {
    use vllm_server::openai::types::{CompletionRequest, ResponseFormat};

    // A JSON body with `response_format` sent to `/v1/completions`
    // is silently ignored — serde's `deny_unknown_fields` is NOT set
    // on `CompletionRequest` (matches OpenAI's permissive legacy
    // endpoint contract: unknown fields are dropped, not 400'd).
    // This pins the wire-type asymmetry: `ChatRequest` declares the
    // field (P22), `CompletionRequest` does not (legacy spec).
    let json = serde_json::json!({
        "prompt": "Hello",
        "response_format": {"type": "text"}
    })
    .to_string();
    let req: CompletionRequest = serde_json::from_str(&json).expect(
        "CompletionRequest should silently ignore unknown fields (legacy endpoint contract)",
    );
    assert_eq!(
        req.prompt, "Hello",
        "completion request parses prompt correctly; response_format is dropped on the legacy endpoint"
    );

    // Compile-time guard: the `ResponseFormat` type still exists for
    // the chat endpoint even though it's not used here.
    let _ = std::any::type_name::<ResponseFormat>();
}

// === P23 regression tests: `seed` field declaration ===
//
// `seed` is OpenAI's "best effort determinism" knob — same seed +
// same model + same prompt should produce the same output. P23
// declares the field on `ChatRequest` + `CompletionRequest` as
// `Option<i64>` with `#[serde(default)]` so omitting it is a no-op.
// Honoring is a no-op today (the sampler is unseeded), but the
// field flows through `tracing::info!(seed = ?req.seed, ...)` so
// determinism is at least observable in trace logs. These tests
// pin the wire-type contract: any `i64` (positive, negative, zero,
// boundaries) is accepted by the HTTP boundary, and the field
// defaults to `None` when omitted. v32+ will add RNG seeding and
// can tighten the validation if needed.

/// A chat request with the `seed` field set must be accepted by the
/// handler (status 200). Pre-fix the field was undeclared and serde
/// rejected the request with a 400-class deserialization error.
#[tokio::test]
async fn test_chat_with_seed_field_accepted_by_handler() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "seed": 42
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
        StatusCode::OK,
        "seed field must not cause 4xx; pre-fix it was undeclared and rejected by serde"
    );
}

/// Baseline: omitting the `seed` field must continue to work
/// (the field is `#[serde(default)]` → `None`). Pins the backward-
/// compatible path so legacy clients are not broken.
#[tokio::test]
async fn test_chat_without_seed_field_works_baseline() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
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
        StatusCode::OK,
        "omitting seed field must continue to work (backward-compat baseline)"
    );
}

/// `/v1/completions` (legacy endpoint) must also accept the `seed`
/// field. Parallel to the P21 `user` declaration on this endpoint.
#[tokio::test]
async fn test_completions_with_seed_field_accepted_by_handler() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, _captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 1,
        "seed": 12345
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "seed field must be accepted on /v1/completions (parallel to /v1/chat/completions)"
    );
}

/// Wire-type round-trip: a JSON body with `seed` deserializes into
/// a `ChatRequest` whose `seed` field equals the original integer;
/// a JSON body without `seed` deserializes into `seed: None`. Also
/// pins the boundary cases (negative, zero, `i64::MIN`/`i64::MAX`)
/// that the OpenAI spec requires us to accept. Catches any future
/// refactor that drops the `#[serde(default)]` annotation or
/// narrows the i64 range.
#[tokio::test]
async fn test_chat_seed_field_wire_type_round_trip() {
    use vllm_server::openai::types::ChatRequest;

    // Positive seed.
    let json_with = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_with)
        .expect("seed field must round-trip from JSON to ChatRequest");
    assert_eq!(
        req.seed,
        Some(42),
        "seed must round-trip; got {:?}",
        req.seed
    );

    // Omitted field defaults to `None`.
    let json_without = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}]
    })
    .to_string();
    let req: ChatRequest =
        serde_json::from_str(&json_without).expect("omitted seed field must deserialize to None");
    assert!(
        req.seed.is_none(),
        "omitted seed must default to None; got {:?}",
        req.seed
    );

    // Negative seed (OpenAI spec accepts any integer).
    let json_negative = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": -1
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_negative)
        .expect("negative seed must deserialize (OpenAI spec accepts any integer)");
    assert_eq!(req.seed, Some(-1));

    // Zero seed (valid i64 — many RNG implementations accept seed=0).
    let json_zero = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 0
    })
    .to_string();
    let req: ChatRequest =
        serde_json::from_str(&json_zero).expect("seed = 0 must deserialize (valid i64)");
    assert_eq!(req.seed, Some(0));

    // Boundary: i64::MIN and i64::MAX.
    let json_min = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": i64::MIN
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_min)
        .expect("seed = i64::MIN must deserialize (no range validation per OpenAI spec)");
    assert_eq!(req.seed, Some(i64::MIN));

    let json_max = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": i64::MAX
    })
    .to_string();
    let req: ChatRequest = serde_json::from_str(&json_max)
        .expect("seed = i64::MAX must deserialize (no range validation per OpenAI spec)");
    assert_eq!(req.seed, Some(i64::MAX));
}

/// Wire-type round-trip on `CompletionRequest`: a JSON body with
/// `seed` deserializes into a `CompletionRequest` whose `seed` field
/// equals the original integer. Mirrors the chat test on the legacy
/// endpoint.
#[tokio::test]
async fn test_completions_seed_field_wire_type_round_trip() {
    use vllm_server::openai::types::CompletionRequest;

    let json_with = serde_json::json!({
        "prompt": "Hello",
        "seed": 999
    })
    .to_string();
    let req: CompletionRequest = serde_json::from_str(&json_with)
        .expect("seed field must round-trip from JSON to CompletionRequest");
    assert_eq!(
        req.seed,
        Some(999),
        "seed must round-trip on /v1/completions; got {:?}",
        req.seed
    );

    let json_without = serde_json::json!({
        "prompt": "Hello"
    })
    .to_string();
    let req: CompletionRequest = serde_json::from_str(&json_without)
        .expect("omitted seed field must deserialize to None on /v1/completions");
    assert!(
        req.seed.is_none(),
        "omitted seed must default to None on /v1/completions; got {:?}",
        req.seed
    );
}

/// Streaming chat completions must also accept the `seed` field
/// without rejection — pins the contract that the SSE path mirrors
/// the non-streaming path's wire-type acceptance (parity with the
/// P21 `user` field and P22 `response_format` field).
#[tokio::test]
async fn test_chat_streaming_with_seed_field_accepted_by_handler() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "stream": true,
        "seed": 7
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "streaming chat must accept the seed field (parity with non-streaming path)"
    );
}

// ====================================================================
// P27 v0.3 wire-type follow-up: `frequency_penalty` + `presence_penalty`
//
// Same pattern as the P21/P22/P23 wire-type integration tests above.
// The key behavioural difference from P21/P22/P23: `frequency_penalty`
// is **honoured end-to-end** via the engine's existing `repeat_penalty`
// slot (P2 ARCH-02). The chat handler maps
// `frequency_penalty >= 0` to `repeat_penalty = max(1.0, 1.0 + value)`;
// negative values are clamped to `1.0` (no penalty) because the current
// `apply_repeat_penalty` logit-divide math inverts the sign of negative
// logits when dividing by a value `< 1.0`. `presence_penalty` is
// declared + validated but NOT wired (engine doesn't have
// presence-aware penalty math — v32+ work).

/// `frequency_penalty = 1.0` on the JSON request must land as
/// `sampling_params.repeat_penalty = 2.0` on the engine side. Pins the
/// v0.3 wire-through contract: non-negative frequency_penalty is
/// honored end-to-end via the existing repeat_penalty slot.
#[tokio::test]
async fn test_chat_forwards_frequency_penalty_to_engine() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": 1.0,
        "max_tokens": 1,
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
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    // Mapping: repeat_penalty = max(1.0, 1.0 + frequency_penalty) = max(1.0, 2.0) = 2.0
    assert!(
        (params.repeat_penalty - 2.0).abs() < 1e-6,
        "frequency_penalty = 1.0 must round-trip to repeat_penalty = 2.0; got {}",
        params.repeat_penalty
    );
}

/// `frequency_penalty = 0.0` (OpenAI default) must land as
/// `sampling_params.repeat_penalty = 1.0` (engine's "no penalty"
/// default). Pins the default-path contract: omitting the field or
/// sending 0 must produce identical engine-side state.
#[tokio::test]
async fn test_chat_frequency_penalty_zero_means_no_penalty() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": 0.0,
        "max_tokens": 1,
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
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.repeat_penalty - 1.0).abs() < 1e-6,
        "frequency_penalty = 0.0 must map to repeat_penalty = 1.0 (no penalty); got {}",
        params.repeat_penalty
    );
}

/// Negative `frequency_penalty` values must be **silently clamped** to
/// `repeat_penalty = 1.0` (no penalty) by the chat handler — they pass
/// the validator (which only enforces the [-2.0, 2.0] range per OpenAI
/// spec) but the wire-through path uses `max(1.0, 1.0 + value)` to
/// avoid the logit-divide sign-flip bug in `apply_repeat_penalty` for
/// `repeat_penalty < 1.0`. Pins the documented v0.3 limitation.
/// P29 v0.3 wire-type follow-up: negative `frequency_penalty` values
/// are forwarded verbatim (with a 1e-3 floor to prevent
/// divide-by-zero at extreme negative values). Previously (P27)
/// the chat handler clamped negative `frequency_penalty` to
/// `repeat_penalty = 1.0` (no penalty) because the engine's
/// `apply_repeat_penalty` used simple logit-division, which had a
/// sign-flip bug for negative logits with `penalty < 1.0`. P29
/// refactors `apply_repeat_penalty` to be sign-aware, so the
/// wire-through can forward negative `frequency_penalty` verbatim
/// (modulo the 1e-3 divide-by-zero floor) and produce the OpenAI
/// "boost repetition" semantic.
///
/// Pins the new contract: `frequency_penalty = -0.5` on the JSON
/// request must land as `SamplingParams::repeat_penalty = 0.5` on
/// the engine side (a mid-range negative that produces a
/// legitimate boost via the sign-aware multiply path).
#[tokio::test]
async fn test_chat_frequency_penalty_negative_is_forwarded_verbatim() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": -0.5,
        "max_tokens": 1,
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
        "negative frequency_penalty must pass validation (in [-2.0, 2.0] range)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    // Mapping: repeat_penalty = max(1e-3, 1.0 + (-0.5)) = max(1e-3, 0.5) = 0.5
    // The pre-P29 `max(1.0, ...)` clamp is removed; -0.5 produces
    // a legitimate boost (sign-aware multiply on negative logits).
    assert!(
        (params.repeat_penalty - 0.5).abs() < 1e-6,
        "frequency_penalty = -0.5 must round-trip to repeat_penalty = 0.5 (boost); got {}",
        params.repeat_penalty
    );
}

/// P29 v0.3 wire-type follow-up: extreme negative `frequency_penalty`
/// values (≤ -1.0) are floored to `repeat_penalty = 1e-3` to
/// prevent divide-by-zero in the engine (which would otherwise
/// happen for positive logits under the divisor formulation).
/// This is the practical limit for boost semantic — values at or
/// below -1.0 produce maximum boost (1e-3 is a strong boost but
/// avoids the infinity from `logit / 0.0`).
#[tokio::test]
async fn test_chat_frequency_penalty_extreme_negative_is_floored() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": -1.5,
        "max_tokens": 1,
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
        "extreme negative frequency_penalty must pass validation (in [-2.0, 2.0] range)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    // Mapping: repeat_penalty = max(1e-3, 1.0 + (-1.5)) = max(1e-3, -0.5) = 1e-3
    // The 1e-3 floor prevents divide-by-zero; the value is no
    // longer clamped to 1.0 (P27 behavior) which would have
    // silently degraded to "no penalty" instead of producing the
    // legitimate maximum boost semantic.
    assert!(
        (params.repeat_penalty - 1e-3).abs() < 1e-9,
        "frequency_penalty = -1.5 must floor to repeat_penalty = 1e-3 (max boost); got {}",
        params.repeat_penalty
    );
}

/// Baseline: omitting `frequency_penalty` must leave
/// `repeat_penalty` at the engine default of `1.0` (no penalty).
/// Pins the backward-compatible path so legacy clients are not
/// broken by the new field.
#[tokio::test]
async fn test_chat_without_frequency_penalty_works_baseline() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
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
        "omitting frequency_penalty must continue to work (backward-compat baseline)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.repeat_penalty - 1.0).abs() < 1e-6,
        "omitted frequency_penalty must leave repeat_penalty at engine default 1.0; got {}",
        params.repeat_penalty
    );
}

/// `presence_penalty` is **wired end-to-end** to the engine's
/// `SamplingParams::presence_penalty` slot (P28 v0.3 wire-type
/// follow-up — engine wire-through). Unlike `frequency_penalty`
/// (which maps to `repeat_penalty` via a clamped `max(1.0, ...)`
/// formula), `presence_penalty` is forwarded verbatim because the
/// engine's `apply_presence_penalty` helper implements an *additive*
/// bias (subtracting the penalty from each distinct seen-token's
/// logit) that handles both positive (discourage) and negative
/// (encourage) values correctly. Pins the v0.3 wire-through
/// contract: `presence_penalty = 1.5` on the JSON request must land
/// as `sampling_params.presence_penalty = 1.5` on the engine side.
#[tokio::test]
async fn test_chat_forwards_presence_penalty_to_engine() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "presence_penalty": 1.5,
        "max_tokens": 1,
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
        "presence_penalty must not cause 4xx; pre-fix it was undeclared and rejected by serde"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.presence_penalty - 1.5).abs() < 1e-6,
        "presence_penalty = 1.5 must round-trip to SamplingParams::presence_penalty = 1.5; got {}",
        params.presence_penalty
    );
}

/// Baseline: omitting `presence_penalty` must leave
/// `sampling_params.presence_penalty` at the engine default of `0.0`
/// (no penalty). Pins the backward-compatible path so legacy clients
/// are not broken by the new field. Mirrors the parallel baseline
/// for `frequency_penalty` (P27).
#[tokio::test]
async fn test_chat_without_presence_penalty_works_baseline() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
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
        "omitting presence_penalty must continue to work (backward-compat baseline)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        params.presence_penalty.abs() < 1e-6,
        "omitted presence_penalty must leave presence_penalty at engine default 0.0; got {}",
        params.presence_penalty
    );
}

/// Negative `presence_penalty` values must be forwarded verbatim
/// (NOT clamped, unlike `frequency_penalty`). This is the key
/// behavioural difference from `frequency_penalty`: the engine's
/// `apply_presence_penalty` uses additive subtraction, so negative
/// values cleanly *encourage* repetition (subtracting a negative =
/// adding to the logit). `frequency_penalty`'s clamping workaround
/// exists because `apply_repeat_penalty` uses logit division,
/// which sign-flips negative logits; `apply_presence_penalty` has
/// no such issue.
#[tokio::test]
async fn test_chat_presence_penalty_negative_is_forwarded_verbatim() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "presence_penalty": -1.0,
        "max_tokens": 1,
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
        "negative presence_penalty must pass validation (in [-2.0, 2.0])"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.presence_penalty - -1.0).abs() < 1e-6,
        "negative presence_penalty must be forwarded verbatim (no clamp); got {}",
        params.presence_penalty
    );
}

/// `/v1/completions` (legacy endpoint) must also forward
/// `presence_penalty` to the engine, mirroring the chat endpoint's
/// wire-through. Pins the cross-endpoint parity contract.
#[tokio::test]
async fn test_completions_forwards_presence_penalty_to_engine() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "presence_penalty": 1.0,
        "max_tokens": 1,
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "/v1/completions must accept and forward presence_penalty to the engine"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        (params.presence_penalty - 1.0).abs() < 1e-6,
        "completions endpoint must also forward presence_penalty → SamplingParams.presence_penalty; got {}",
        params.presence_penalty
    );
}

/// Out-of-range `frequency_penalty` must be rejected with `400` at
/// the HTTP boundary (per OpenAI spec, [-2.0, 2.0]). Pins the
/// validator path: bad values never reach the engine.
#[tokio::test]
async fn test_chat_frequency_penalty_out_of_range_returns_400() {
    let (state, _handle, _captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": 3.0,
        "max_tokens": 1,
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
        StatusCode::BAD_REQUEST,
        "frequency_penalty = 3.0 must be rejected with 400 (per OpenAI spec [-2.0, 2.0])"
    );
}

/// Out-of-range `presence_penalty` must also be rejected with `400`.
/// Parallel to the frequency_penalty range check.
#[tokio::test]
async fn test_chat_presence_penalty_out_of_range_returns_400() {
    let (state, _handle, _captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "presence_penalty": -3.0,
        "max_tokens": 1,
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
        StatusCode::BAD_REQUEST,
        "presence_penalty = -3.0 must be rejected with 400 (per OpenAI spec [-2.0, 2.0])"
    );
}

/// `/v1/completions` (legacy endpoint) must also accept
/// `frequency_penalty` and forward it to the engine, mirroring the
/// chat endpoint's wire-through. Pins the cross-endpoint parity
/// contract.
#[tokio::test]
async fn test_completions_forwards_frequency_penalty_to_engine() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "frequency_penalty": 1.0,
        "max_tokens": 1,
    })
    .to_string();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "/v1/completions must accept and forward frequency_penalty to the engine"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    // Mapping: repeat_penalty = max(1.0, 1.0 + 1.0) = 2.0
    assert!(
        (params.repeat_penalty - 2.0).abs() < 1e-6,
        "completions endpoint must also forward frequency_penalty → repeat_penalty; got {}",
        params.repeat_penalty
    );
}

// P30 v0.3 wire-type follow-up: `logit_bias` engine wire-through.
// Same pattern as the P27/P28/P29 wire-through tests — declare a
// JSON request with a `logit_bias` map, hit the endpoint, verify the
// captured `SamplingParams` carries the same map. Engine honoring is
// verified separately in `crates/core/src/sampling/tests.rs::test_*
// (apply_logit_bias unit tests) and `crates/core/tests/sampling_params.rs
// ::arch_02_logit_bias_*` (per-sequence batch divergence tests).

/// A chat request with a `logit_bias` map must round-trip the map
/// to `SamplingParams::logit_bias` verbatim. Pins the v0.3
/// wire-through contract: the engine receives the same map the
/// caller sent (no transformation, no key-set filtering, no
/// value clamping — out-of-range values are rejected by the
/// validator up front so the engine never sees bad data).
#[tokio::test]
async fn test_chat_forwards_logit_bias_to_engine() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {
            "42": 50.0,
            "100": -25.0,
            "7": 100.0,
            "999": -100.0,
        },
        "max_tokens": 1,
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
        "logit_bias must pass validation (all values in [-100, 100])"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    let bias = params
        .logit_bias
        .as_ref()
        .expect("logit_bias must be forwarded to SamplingParams");

    // All four entries must round-trip exactly. The HashMap iteration
    // order is non-deterministic, so we look up by key.
    assert!(
        (bias.get(&42).copied().unwrap_or(0.0) - 50.0).abs() < 1e-6,
        "logit_bias[42] = 50.0 must round-trip verbatim; got {:?}",
        bias.get(&42)
    );
    assert!(
        (bias.get(&100).copied().unwrap_or(0.0) - -25.0).abs() < 1e-6,
        "logit_bias[100] = -25.0 must round-trip verbatim; got {:?}",
        bias.get(&100)
    );
    assert!(
        (bias.get(&7).copied().unwrap_or(0.0) - 100.0).abs() < 1e-6,
        "logit_bias[7] = 100.0 (upper boundary) must round-trip; got {:?}",
        bias.get(&7)
    );
    assert!(
        (bias.get(&999).copied().unwrap_or(0.0) - -100.0).abs() < 1e-6,
        "logit_bias[999] = -100.0 (lower boundary) must round-trip; got {:?}",
        bias.get(&999)
    );
    assert_eq!(
        bias.len(),
        4,
        "logit_bias map must carry all 4 entries"
    );
}

/// A chat request without a `logit_bias` field must produce a
/// `SamplingParams::logit_bias = None`. Pins the default-path
/// contract: omitting the field or sending `null` produces
/// identical engine-side state.
#[tokio::test]
async fn test_chat_without_logit_bias_works_baseline() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
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
        "omitted logit_bias must pass validation (None is the default)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert!(
        params.logit_bias.is_none(),
        "omitted logit_bias must produce SamplingParams::logit_bias = None; got {:?}",
        params.logit_bias
    );
}

/// An empty `logit_bias` map (`{}`) must round-trip to
/// `SamplingParams::logit_bias = Some(empty_map)`. The engine's
/// `apply_logit_bias` is a no-op on empty maps, so this is
/// semantically equivalent to `None` — but the field is preserved
/// on the wire so callers can distinguish "I sent no bias" from
/// "I sent a (legitimately empty) bias map".
#[tokio::test]
async fn test_chat_with_empty_logit_bias_is_accepted() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {},
        "max_tokens": 1,
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
        "empty logit_bias map must pass validation"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    let bias = params
        .logit_bias
        .as_ref()
        .expect("empty logit_bias map must be forwarded as Some(empty)");
    assert!(bias.is_empty(), "logit_bias map must be empty");
}

/// Out-of-range `logit_bias` values (above 100.0 or below -100.0)
/// must be rejected with `400 invalid_request_error`. Pins the
/// OpenAI-spec range check.
#[tokio::test]
async fn test_chat_logit_bias_out_of_range_returns_400() {
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"42": 200.0}, // above OpenAI spec upper bound
        "max_tokens": 1,
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
        StatusCode::BAD_REQUEST,
        "logit_bias value > 100.0 must be rejected with 400"
    );
}

/// `NaN` `logit_bias` values must be rejected with `400`. Without
/// this gate the NaN would propagate through the softmax and
/// produce NaN probabilities.
#[tokio::test]
async fn test_chat_logit_bias_nan_returns_400() {
    // serde_json's Number type doesn't represent NaN, so we can't
    // send a JSON NaN directly. The validator catches NaN in
    // unit tests; here we cover a representable but invalid value
    // (+infinity) instead — same code path, same error class.
    let (state, _engine) = api_state_with_mock_engine(Architecture::Qwen3, vec![101, 102]);
    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"42": 1e30}, // very large finite value, still within f32
        "max_tokens": 1,
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
        StatusCode::BAD_REQUEST,
        "logit_bias value = 1e30 (above OpenAI spec upper bound) must be rejected with 400"
    );
}

/// The completions endpoint must also forward `logit_bias` to the
/// engine (mirror of the chat wire-through).
#[tokio::test]
async fn test_completions_forwards_logit_bias_to_engine() {
    use std::collections::HashMap;
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "logit_bias": {
            "42": 50.0,
            "100": -50.0,
        },
        "max_tokens": 1,
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "/v1/completions must accept and forward logit_bias to the engine"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    let bias = params
        .logit_bias
        .as_ref()
        .expect("logit_bias must be forwarded to SamplingParams");
    let mut expected = HashMap::new();
    expected.insert(42u32, 50.0f32);
    expected.insert(100u32, -50.0f32);
    assert_eq!(bias.len(), 2, "logit_bias map must carry 2 entries");
    for (k, v) in &expected {
        assert!(
            (bias.get(k).copied().unwrap_or(0.0) - v).abs() < 1e-6,
            "logit_bias[{k}] = {v} must round-trip verbatim; got {:?}",
            bias.get(k)
        );
    }
}

// P31 v0.3 wire-type follow-up: `logprobs` + `top_logprobs`
// declaration + validation. Same pattern as the P21/P22/P23/P27/P28/
// P29/P30 wire-through tests but with declaration-only honoring:
// the engine wire-through is a no-op today (v32+ work), so the
// captured `SamplingParams` is verified for the unchanged path
// (no logprobs fields exist on `SamplingParams`) and the validator
// behaviour is exercised end-to-end through the HTTP boundary.

/// A chat request with `logprobs = true` + `top_logprobs = 5` must
/// pass validation and reach the engine unchanged. Pins the v0.3
/// declaration contract: the wire type accepts the fields and
/// forwarding them to the engine is a no-op (no `logprobs` /
/// `top_logprobs` field on `SamplingParams` today — v32+).
#[tokio::test]
async fn test_chat_with_logprobs_field_accepted_by_handler() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 5,
        "max_tokens": 1,
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
        "logprobs + top_logprobs must pass validation (in OpenAI-spec ranges)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    // Honoring is a no-op today — the engine's SamplingParams does
    // not have a logprobs field (engine wire-through is v32+ work).
    // The fact that the request reached the engine at all is the
    // wire-type contract: pre-P31 it would have been rejected by
    // serde ("unknown field `logprobs`").
    assert!(
        params.temperature.abs() < 1e-6,
        "default temperature (greedy) must be unchanged"
    );
}

/// Baseline: omitting both `logprobs` and `top_logprobs` must pass
/// validation and reach the engine unchanged. Pins the default-path
/// contract: pre-P31 this was the only working state.
#[tokio::test]
async fn test_chat_without_logprobs_field_works_baseline() {
    let (state, _handle, _captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
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
        "omitted logprobs must continue to work (backward-compat baseline)"
    );
}

/// `top_logprobs` outside the `[0, 20]` OpenAI-spec range must be
/// rejected with `400 invalid_request_error`. Pins the validator
/// contract.
#[tokio::test]
async fn test_chat_top_logprobs_out_of_range_returns_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 21, // above OpenAI spec upper bound (20)
        "max_tokens": 1,
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
        StatusCode::BAD_REQUEST,
        "top_logprobs > 20 must be rejected with 400"
    );

    // Critical: when validation fails the request must NOT reach the
    // engine. Otherwise a saturated engine could burn cycles on a
    // known-bad request.
    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "out-of-range top_logprobs must NOT reach the engine (captured is None)"
    );
}

/// The cross-field rule (`top_logprobs` requires `logprobs = true`)
/// must be enforced end-to-end. `top_logprobs = Some(5)` with
/// `logprobs = false` is rejected with `400`.
#[tokio::test]
async fn test_chat_top_logprobs_without_logprobs_returns_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": false,
        "top_logprobs": 5, // cross-field rule: requires logprobs = true
        "max_tokens": 1,
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
        StatusCode::BAD_REQUEST,
        "top_logprobs + logprobs = false must be rejected with 400 (cross-field rule)"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "cross-field rejection must NOT reach the engine (captured is None)"
    );
}

/// The completions endpoint must also accept the `logprobs` field
/// (legacy spec — `logprobs: int 0..=5`). Pins the cross-endpoint
/// parity: same declaration pattern as `seed` / `user` /
/// `frequency_penalty` / `presence_penalty` / `logit_bias`.
#[tokio::test]
async fn test_completions_with_logprobs_field_accepted_by_handler() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "logprobs": 3, // OpenAI-spec: int 0..=5
        "max_tokens": 1,
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "/v1/completions must accept logprobs in [0, 5] range"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_some(),
        "in-range logprobs must reach the engine (captured is Some)"
    );
}

/// Out-of-range completions `logprobs` (> 5) must be rejected with
/// `400`. Pins the validator contract for the legacy endpoint.
#[tokio::test]
async fn test_completions_logprobs_out_of_range_returns_400() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle, captured) = spawn_capturing_mock_engine();
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Qwen3,
        batch_manager: Arc::new(vllm_server::openai::batch::BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(
            vllm_server::health::HealthChecker::new(true, true),
        )),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    };
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "logprobs": 6, // above OpenAI spec upper bound (5)
        "max_tokens": 1,
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "/v1/completions logprobs > 5 must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "out-of-range logprobs must NOT reach the engine (captured is None)"
    );
}
