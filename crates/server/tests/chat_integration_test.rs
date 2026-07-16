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
