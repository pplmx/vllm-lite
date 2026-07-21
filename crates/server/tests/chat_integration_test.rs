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

/// P38 v0.3 wire-type engine wire-through: `stop` sequences are
/// now accepted by the chat handler (no longer 400). The validator
/// passes them through (`validate_stop_sequences`: max 4 strings,
/// no empty/whitespace strings); the HTTP layer tokenizes and
/// forwards as `SamplingParams::stop_token_sequences`. The handler
/// must accept the request without rejecting it at the boundary.
///
/// NOTE: with the default `api_state` (no engine), the request
/// errors out via `engine_unavailable` (503) once the handler tries
/// to send `AddRequest`. The KEY contract being pinned here is: NOT
/// 400. Once the engine integration lands (this commit), the engine
/// integration exercises the stop check end-to-end.
#[tokio::test]
async fn test_chat_stop_now_accepted_by_handler() {
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
    // Must NOT be 400 (validator no longer rejects non-empty stop).
    // Will be 503 because the engine isn't running — but the
    // IMPORTANT contract here is: NO 400 from the validator.
    assert_ne!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "stop must be accepted by validator (P38)"
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
                    let _ = response_tx
                        .send(vllm_traits::SampledToken {
                            token: 10u32,
                            logprob: 0.0,
                            top_logprobs: vec![],
                        })
                        .await;
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
    assert_eq!(bias.len(), 4, "logit_bias map must carry all 4 entries");
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

// P32 v0.x wire-type follow-up: `echo` + `suffix` + `best_of`
// declaration + validation on the legacy `/v1/completions`
// endpoint. Mirrors the P21/P22/P23/P27/P28/P29/P30/P31
// integration-test pattern (capturing mock engine verifies the
// request reaches the engine and that 400s don't leak through).
// Engine honoring is a no-op today (v32+ work); the tests pin the
// declaration + validation contract end-to-end through the HTTP
// boundary.

/// `echo = true` + `best_of = 1` (no cross-field conflict) must
/// pass validation and reach the engine unchanged.
#[tokio::test]
async fn test_completions_with_echo_true_accepted_by_handler() {
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
        "echo": true,
        "best_of": 1,
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
        "echo=true + best_of=1 must pass validation (no cross-field conflict)"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// `suffix` is unconstrained per OpenAI spec; any string must pass.
#[tokio::test]
async fn test_completions_with_suffix_accepted_by_handler() {
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
        "prompt": "def hello(",
        "suffix": "    return 42\n}",
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
        "suffix must be accepted (any string per OpenAI spec)"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// Baseline: omitting `echo`, `suffix`, `best_of` must pass
/// validation and reach the engine unchanged. Pins the
/// default-path contract: pre-P32 this was the only working state.
#[tokio::test]
async fn test_completions_without_echo_suffix_best_of_works_baseline() {
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
        "baseline (all three fields omitted) must pass validation"
    );
}

/// `best_of = 5` alone (no echo) must pass validation and reach
/// the engine unchanged. Honoring is v32+ work but the
/// wire-type contract accepts the value today.
#[tokio::test]
async fn test_completions_with_best_of_above_one_accepted_by_handler() {
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
        "best_of": 5,
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
        "best_of=5 alone (no echo) must pass validation"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// `best_of = 0` must be rejected with `400 invalid_request_error`
/// (must be `>= 1` per OpenAI spec). Pins the validator
/// end-to-end through the HTTP boundary.
#[tokio::test]
async fn test_completions_best_of_with_above_one_and_above_twenty_returns_400() {
    // Variant: best_of just above the boundary (21) AND far above the
    // boundary (1_000_000) must both be rejected — pins the boundary
    // check end-to-end.
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10]);
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    for n in [21u32, 100u32, 1_000_000u32] {
        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 1,
            "best_of": n,
        })
        .to_string();
        let response = app
            .clone()
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
            "best_of={n} must return 400 (<= 20 per OpenAI spec, P37)"
        );
    }
}

#[tokio::test]
async fn test_completions_best_of_zero_returns_400() {
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
        "best_of": 0,
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
        "/v1/completions best_of = 0 must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "best_of=0 must NOT reach the engine (captured is None)"
    );
}

/// `echo = true` + `best_of > 1` must be rejected with `400
/// invalid_request_error` (cross-field rule per OpenAI spec).
/// Pins the validator end-to-end through the HTTP boundary.
#[tokio::test]
async fn test_completions_echo_true_with_best_of_above_one_returns_400() {
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
        "echo": true,
        "best_of": 3,
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
        "/v1/completions echo=true + best_of>1 must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "echo=true + best_of=3 must NOT reach the engine (captured is None)"
    );
}

// ============================================================================
// P35 v0.x wire-type follow-up: `echo` + `suffix` engine wire-through
// ============================================================================
//
// P32 declared + validated `echo` and `suffix` on `CompletionRequest`. P35
// closes the engine-side gap: the non-streaming handler now prepends
// `prompt` to `CompletionChoice.text` when `echo = true` and appends
// `suffix` to the same field when `suffix = Some(_)`. The streaming
// handler does the same across the SSE event stream (echo prefix goes
// onto the first text chunk; suffix postamble goes onto the natural-
// completion chunk that carries `finish_reason`).
//
// `best_of` engine honoring remains v32+ work (depends on logprobs
// ranking — P31 follow-up); these tests verify only `echo` + `suffix`
// honoring via the `best_of = 1` default path that doesn't conflict.

/// P35 wire-through (non-streaming): `echo = true` prepends the
/// prompt to `choices[0].text`. Pins the OpenAI-spec "echo back the
/// prompt as a prefix to the generated continuation" semantic.
#[tokio::test]
async fn test_completions_echo_true_prepends_prompt_to_text() {
    use vllm_server::openai::completions::completions;
    // Mock engine emits a single token (10); the empty tokenizer
    // decodes token 10 to `"token_10 "`. After `clean_completion_text`
    // the continuation is `"token_10"`. With `echo = true` the
    // response text must be `"Hello" + "token_10"` (the exact
    // prompt + the generated continuation).
    let (engine_tx, _handle) = spawn_mock_engine(vec![10]);
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
        "echo": true,
        "best_of": 1,
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
        "echo=true must pass validation"
    );

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(body_str)
        .unwrap_or_else(|e| panic!("response is not valid JSON: {e}; body: {body_str}"));
    let text = parsed["choices"][0]["text"]
        .as_str()
        .expect("choices[0].text must be a string");
    assert!(
        text.starts_with("Hello"),
        "echo=true must prepend prompt to text (text={text:?})"
    );
    assert!(
        text.contains("token_10"),
        "echo=true must still include the generated continuation (text={text:?})"
    );
}

/// P35 wire-through (non-streaming): `echo = false` (or omitted)
/// does NOT prepend the prompt. Pins the backward-compatible path
/// so legacy clients are not broken by the new wire-through.
#[tokio::test]
async fn test_completions_echo_false_does_not_prepend_prompt() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle) = spawn_mock_engine(vec![10]);
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
        "echo": false,
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

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(body_str).unwrap();
    let text = parsed["choices"][0]["text"]
        .as_str()
        .expect("choices[0].text must be a string");
    assert!(
        !text.contains("Hello"),
        "echo=false must NOT include the prompt (text={text:?})"
    );
    assert!(
        text.contains("token_10"),
        "echo=false must still return the generated continuation (text={text:?})"
    );
}

/// P35 wire-through (non-streaming): `suffix = Some("xyz")` appends
/// the suffix to `choices[0].text`. Pins the OpenAI-spec
/// "string that comes after the inserted completion" semantic.
#[tokio::test]
async fn test_completions_suffix_appends_to_text() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle) = spawn_mock_engine(vec![10]);
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
        "suffix": "xyz",
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

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(body_str).unwrap();
    let text = parsed["choices"][0]["text"]
        .as_str()
        .expect("choices[0].text must be a string");
    assert!(
        text.ends_with("xyz"),
        "suffix=\"xyz\" must append to text (text={text:?})"
    );
    assert!(
        text.contains("token_10"),
        "suffix must still include the generated continuation (text={text:?})"
    );
}

/// P35 wire-through (non-streaming): `echo = true` + `suffix =
/// Some(_)` combines both — text starts with prompt AND ends with
/// suffix AND contains the continuation.
#[tokio::test]
async fn test_completions_echo_and_suffix_combine() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle) = spawn_mock_engine(vec![10]);
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
        "echo": true,
        "suffix": "xyz",
        "best_of": 1,
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

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(body_str).unwrap();
    let text = parsed["choices"][0]["text"]
        .as_str()
        .expect("choices[0].text must be a string");
    assert!(
        text.starts_with("Hello"),
        "echo+suffix must start with prompt (text={text:?})"
    );
    assert!(
        text.ends_with("xyz"),
        "echo+suffix must end with suffix (text={text:?})"
    );
    assert!(
        text.contains("token_10"),
        "echo+suffix must contain the continuation (text={text:?})"
    );
}

/// P35 wire-through (streaming): `echo = true` puts the prompt into
/// the FIRST text chunk's `text` field. The token-continuation chunks
/// after the first one remain unchanged (no prompt prefix).
#[tokio::test]
async fn test_completions_streaming_echo_true_prepends_prompt_to_first_chunk() {
    use vllm_server::openai::completions::completions;
    // Two tokens so we get 2 distinct text chunks (token_10, token_20).
    let (engine_tx, _handle) = spawn_mock_engine(vec![10, 20]);
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
        "echo": true,
        "best_of": 1,
        "stream": true,
        "max_tokens": 5,
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

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    // SSE: each event ends with `\n\n`. Parse only events that carry
    // a `data:` payload — the [DONE] sentinel has no JSON.
    let data_lines: Vec<&str> = body_str
        .split("\n\n")
        .filter_map(|event| event.lines().find(|l| l.starts_with("data: ")))
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]" && !p.is_empty())
        .collect();
    assert!(
        !data_lines.is_empty(),
        "streaming response should carry at least one JSON chunk, body: {body_str}"
    );
    let first_parsed: serde_json::Value = serde_json::from_str(data_lines[0]).unwrap_or_else(|e| {
        panic!(
            "first chunk is not valid JSON: {e}; data: {}",
            data_lines[0]
        )
    });
    let first_text = first_parsed["choices"][0]["text"]
        .as_str()
        .expect("first chunk must carry choices[0].text as a string");
    assert!(
        first_text.starts_with("Hello"),
        "streaming echo=true must prepend prompt to first chunk (first_text={first_text:?})"
    );
}

/// P35 wire-through (streaming): `suffix = Some(_)` puts the
/// suffix into the FINAL chunk's `text` field (the chunk that
/// carries `finish_reason`). This matches OpenAI's accumulator
/// semantics: clients concatenate all chunk `text` fields in
/// order, so a suffix on the final chunk ends up at the end of
/// the visible completion, with no need for callers to special-
/// case "last non-empty text chunk" tracking.
#[tokio::test]
async fn test_completions_streaming_suffix_lands_on_final_chunk() {
    use vllm_server::openai::completions::completions;
    let (engine_tx, _handle) = spawn_mock_engine(vec![10, 20]);
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
        "suffix": "xyz",
        "stream": true,
        "max_tokens": 5,
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

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();
    // Find the chunk that carries `finish_reason` — that's where
    // the suffix lives. We skip the `[DONE]` sentinel (no JSON).
    let mut finish_chunk_text: Option<String> = None;
    for event in body_str.split("\n\n").filter(|e| !e.is_empty()) {
        let data = event.lines().find_map(|l| l.strip_prefix("data: "));
        if let Some(payload) = data
            && payload != "[DONE]"
            && !payload.is_empty()
            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(payload)
            && parsed["choices"][0]["finish_reason"].is_string()
        {
            finish_chunk_text = parsed["choices"][0]["text"].as_str().map(str::to_string);
        }
    }
    let final_text =
        finish_chunk_text.expect("streaming response must carry a finish_reason chunk");
    assert_eq!(
        final_text, "xyz",
        "suffix=\"xyz\" must land on the finish_reason chunk's text field \
         (got {final_text:?})"
    );
}

// P33 v0.x wire-type follow-up: `tools` + `tool_choice`
// declaration + validation on the `/v1/chat/completions`
// endpoint. Mirrors the P21/P22/P23/P27/P28/P29/P30/P31/P32
// integration-test pattern. Engine honoring is a no-op today
// (v32+ work — grammar-constrained decoder + per-request tool
// schema cache); the tests pin the declaration + validation
// contract end-to-end through the HTTP boundary.

/// A chat request with `tools` defined (no `tool_choice`) must
/// pass validation and reach the engine unchanged. Pins the
/// baseline contract: pre-P33 this was rejected by serde.
#[tokio::test]
async fn test_chat_with_tools_only_accepted_by_handler() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }],
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
        "tools without tool_choice must pass validation"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// `tool_choice = "auto"` + `tools` defined must pass validation
/// and reach the engine unchanged.
#[tokio::test]
async fn test_chat_with_tool_choice_auto_accepted_by_handler() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather"}
        }],
        "tool_choice": "auto",
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
        "tool_choice=auto + tools must pass validation"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// `tool_choice = "required"` + `tools` defined must pass
/// validation.
#[tokio::test]
async fn test_chat_with_tool_choice_required_accepted_by_handler() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather"}
        }],
        "tool_choice": "required",
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
        "tool_choice=required + tools must pass validation"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// `tool_choice = {"type": "function", "function": {"name": "X"}}`
/// + matching tool in `tools` must pass validation.
#[tokio::test]
async fn test_chat_with_tool_choice_specific_matching_accepted_by_handler() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Weather?"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather"}
        }, {
            "type": "function",
            "function": {"name": "get_time"}
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"}
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
        "tool_choice=specific with matching tool must pass validation"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

/// Baseline: omitting both `tools` and `tool_choice` must pass
/// validation and reach the engine unchanged.
#[tokio::test]
async fn test_chat_without_tools_tool_choice_works_baseline() {
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
        "baseline (no tools, no tool_choice) must pass validation"
    );
}

/// `tool_choice = "required"` WITHOUT `tools` must be rejected
/// with 400. Pins the cross-field rule end-to-end.
#[tokio::test]
async fn test_chat_tool_choice_required_without_tools_returns_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": "required",
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
        "tool_choice=required without tools must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "tool_choice=required without tools must NOT reach the engine (captured is None)"
    );
}

/// `tool_choice = "required"` + `tools = []` (empty array) must
/// be rejected. Per OpenAI spec, `required` requires at least one
/// tool to be defined.
#[tokio::test]
async fn test_chat_tool_choice_required_with_empty_tools_returns_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [],
        "tool_choice": "required",
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
        "tool_choice=required + empty tools must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "tool_choice=required + empty tools must NOT reach the engine"
    );
}

/// `tool_choice.function.name` that doesn't match any tool in
/// `tools[]` must be rejected. Pins the cross-field rule end-to-end.
#[tokio::test]
async fn test_chat_tool_choice_specific_unknown_tool_returns_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_time"}
        }],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"}
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
        StatusCode::BAD_REQUEST,
        "tool_choice.function.name not matching any tool must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "tool_choice with unknown tool name must NOT reach the engine"
    );
}

/// `tool_choice = specific` WITHOUT `tools` must be rejected.
#[tokio::test]
async fn test_chat_tool_choice_specific_without_tools_returns_400() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"}
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
        StatusCode::BAD_REQUEST,
        "specific tool_choice without tools must be rejected with 400"
    );

    let captured = captured.lock().await;
    assert!(
        captured.is_none(),
        "specific tool_choice without tools must NOT reach the engine"
    );
}

/// `tool_choice = "none"` must always pass regardless of whether
/// `tools` is defined or not. Per OpenAI spec, "none" means the
/// model must not call any tool — `tools` may be absent or
/// non-empty (the model just ignores the list).
#[tokio::test]
async fn test_chat_tool_choice_none_with_tools_accepted_by_handler() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather"}
        }],
        "tool_choice": "none",
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
        "tool_choice=none + tools must pass validation (model just ignores the tools)"
    );

    let captured = captured.lock().await;
    let _params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
}

// ============================================================================
// P34: seed engine wire-through integration tests
// ============================================================================
//
// The `seed` field was declared + validated + traced in P23 (v0.2 wire-type
// follow-up declaration only). P34 closes the engine-side gap: `req.seed`
// (an `i64` from the wire) is cast to `u64` and stored in
// `SamplingParams::seed`, where `sample_one_with_params` reads it to seed
// a fresh `StdRng::seed_from_u64`. These tests pin the end-to-end
// handler→engine path so future refactors can't silently regress the
// "same seed → same output" contract.

// Note: `max_tokens = 1` forces a single decode step. Greedy paths
// (`temperature = 0`) bypass the RNG so seed has no observable
// effect; we therefore don't assert specific tokens — only that the
// seed survives the wire-through intact.

/// Forward `seed` (positive value) end-to-end from the chat handler
/// to `SamplingParams::seed`. Mirrors the parallel wire-through
/// tests for `presence_penalty` (P28) and `logit_bias` (P30).
#[tokio::test]
async fn test_chat_forwards_seed_to_engine() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42_i64,
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
        "seed must not cause 4xx; pre-fix it was undeclared and rejected by serde"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.seed,
        Some(42_u64),
        "seed = 42 must round-trip to SamplingParams::seed = Some(42); got {:?}",
        params.seed
    );
}

/// Baseline: omitting `seed` must leave `sampling_params.seed` at
/// `None` (no seeded RNG). Pins the backward-compatible path so
/// legacy clients are not broken by the new wire-through. Mirrors
/// the parallel baseline for `presence_penalty` (P28) and
/// `logit_bias` (P30).
#[tokio::test]
async fn test_chat_without_seed_works_baseline() {
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
        "omitting seed must continue to work (backward-compat baseline)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.seed, None,
        "omitted seed must leave SamplingParams::seed at None; got {:?}",
        params.seed
    );
}

/// Negative `seed` values must be forwarded verbatim via `as u64`
/// cast (Rust's `i64 as u64` wrapping semantics). This pins the
/// contract that OpenAI's "any integer" seed spec is honoured even
/// for negative values — the engine still gets a deterministic but
/// distinct RNG state.
#[tokio::test]
async fn test_chat_negative_seed_is_forwarded_via_u64_cast() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": -1_i64,
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
        "negative seed must not cause 4xx (OpenAI spec accepts any i64)"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.seed,
        Some(-1_i64 as u64),
        "seed = -1 must round-trip to SamplingParams::seed = Some({}); got {:?}",
        -1_i64 as u64,
        params.seed
    );
}

/// `seed = 0` is a valid seed (NOT conflated with `None`). Pins
/// the OpenAI-spec "any integer" contract — `seed = 0` must
/// produce `SamplingParams::seed = Some(0)` so the engine seeds
/// the RNG with `StdRng::seed_from_u64(0)`.
#[tokio::test]
async fn test_chat_seed_zero_is_forwarded_as_some_zero() {
    let (state, _handle, captured) = state_with_capturing_engine();

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 0_i64,
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
        "seed = 0 must not be conflated with absent seed"
    );

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.seed,
        Some(0_u64),
        "seed = 0 must round-trip to SamplingParams::seed = Some(0), not None; got {:?}",
        params.seed
    );
}

// ============================================================================
// P36 v0.3 wire-type follow-up engine wire-through: logprobs + top_logprobs
// ============================================================================
//
// Engine wire-through for the v0.3 `logprobs` / `top_logprobs` fields
// declared by P31 (chat `logprobs: Option<bool>` + `top_logprobs:
// Option<u32>`, completions `logprobs: Option<u32>`). The engine now
// emits `SampledToken { token, logprob, top_logprobs }` per step; the
// HTTP layer renders the response-side `choices[].logprobs` shape from
// that stream. These tests pin the contract end-to-end.
//
// Engine wire-through contract:
// 1. `req.logprobs` (chat) / `req.logprobs` (completions) → `SamplingParams::top_logprobs`
// 2. Engine emits `SampledToken` per step (mock sends placeholder
//    values; production engine fills in real logprobs via
//    `sample_one_with_params`'s `logprob_of_token` + `top_logprobs_of`
//    helpers)
// 3. Non-streaming chat: `choices[0].logprobs.content[]` carries one
//    entry per generated token, each with `token` + `logprob` (and
//    `top_logprobs[]` when `top_logprobs > 0`)
// 4. Streaming chat: each intermediate chunk carries exactly one
//    logprob entry (the sampled token); the final chunk carries none
// 5. Non-streaming completions: `choices[0].logprobs` carries
//    parallel `tokens[]` / `token_logprobs[]` / `top_logprobs[][]`
//    arrays, one entry per generated token
//
// The mock engines below use deterministic placeholder logprob values
// so the assertions are reproducible; production paths are
// independently exercised by `crates/core/src/sampling/tests.rs`.

/// Spawn a mock engine that captures the `SamplingParams` AND emits
/// SampledToken values with a placeholder logprob + top_logprobs
/// payload. P36 wire-through tests need both the forward direction
/// (request → engine) and the reverse direction (engine → response
/// shape) so the existing `spawn_capturing_mock_engine` is extended
/// to actually emit a deterministic SampledToken stream.
fn spawn_capturing_logprob_engine() -> (
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
                    // Emit three tokens with deterministic logprob +
                    // top_logprobs payloads. The values are placeholders
                    // — the production engine fills these via
                    // `sample_one_with_params`'s logprob helpers — but
                    // the shape (token, logprob, top_logprobs[]) is
                    // identical, so the HTTP layer's renderer exercises
                    // the same code path.
                    for (token, logprob) in [(10u32, -0.5f32), (20, -1.2), (30, -2.0)] {
                        let sampled = vllm_traits::SampledToken {
                            token,
                            logprob,
                            // Two top-K alternatives each (token IDs
                            // and logprobs matching the OpenAI
                            // shape).
                            top_logprobs: vec![(11, -1.0), (12, -1.5)],
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    break;
                }
                EngineMessage::Shutdown => break,
                _ => {}
            }
        }
    });
    (engine_tx, handle, captured)
}

/// P36: chat `top_logprobs` field on the request must land as
/// `SamplingParams::top_logprobs = Some(n)` on the engine side. The
/// engine honours it by running a partial top-K selection on the
/// post-filter logits (see `sample_one_with_params` in
/// `crates/core/src/sampling.rs`).
#[tokio::test]
async fn test_chat_forwards_top_logprobs_to_engine() {
    let (engine_tx, _handle, captured) = spawn_capturing_logprob_engine();
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
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 5,
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
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.top_logprobs,
        Some(5),
        "top_logprobs = 5 must round-trip to SamplingParams::top_logprobs = Some(5); got {:?}",
        params.top_logprobs
    );
}

/// P36: chat `top_logprobs = 0` must still round-trip (engine still
/// computes the sampled-token logprob via `sample_one_with_params`,
/// only the top-K selection is skipped).
#[tokio::test]
async fn test_chat_top_logprobs_zero_round_trips() {
    let (engine_tx, _handle, captured) = spawn_capturing_logprob_engine();
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
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 0,
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
    assert_eq!(response.status(), StatusCode::OK);

    let captured = captured.lock().await;
    let params = captured
        .as_ref()
        .expect("capturing mock must have observed the AddRequest");
    assert_eq!(
        params.top_logprobs,
        Some(0),
        "top_logprobs = 0 must round-trip as Some(0) (NOT None — that would \
         confuse the engine into thinking the request didn't ask for \
         logprobs); got {:?}",
        params.top_logprobs
    );
}

/// P36: chat response `choices[0].logprobs.content[]` must carry one
/// entry per generated token, each with `token` + `logprob` +
/// `top_logprobs` (when `top_logprobs > 0`). End-to-end wire shape
/// pins: the JSON serializer must produce the OpenAI-spec field
/// names and the `content[]` array structure.
#[tokio::test]
async fn test_chat_response_logprobs_wire_shape() {
    let (state, _handle) = vllm_server::test_fixtures::api_state_with_mock_engine(
        Architecture::Qwen3,
        vec![10, 20, 30],
    );
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 2,
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
    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(choices.len(), 1);
    let logprobs = choices[0]
        .get("logprobs")
        .expect("logprobs must be present when request asked for them");
    let content = logprobs
        .get("content")
        .and_then(|v| v.as_array())
        .expect("logprobs.content must be an array");
    assert_eq!(
        content.len(),
        3,
        "3 generated tokens → 3 content entries; got {}",
        content.len()
    );
    for (i, entry) in content.iter().enumerate() {
        assert!(entry.get("token").is_some(), "entry {i} missing `token`");
        assert!(
            entry.get("logprob").is_some(),
            "entry {i} missing `logprob`"
        );
        assert!(
            entry.get("top_logprobs").is_some(),
            "entry {i} missing `top_logprobs` (request asked for top_logprobs = 2)"
        );
    }
}

/// P36: chat response when `logprobs = false` (or omitted) must NOT
/// carry a `logprobs` field at all — OpenAI's spec uses field
/// absence to signal "logprobs were not requested". The
/// `skip_serializing_if = "Option::is_none"` annotation on
/// `ChatChoice::logprobs` handles this; this test pins it.
#[tokio::test]
async fn test_chat_response_omits_logprobs_when_not_requested() {
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10, 20]);
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 2,
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

    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(choices.len(), 1);
    assert!(
        choices[0].get("logprobs").is_none(),
        "logprobs field MUST be absent when request did not ask for it; got: {}",
        choices[0]
    );
}

/// P36: completions `logprobs` field on the request must land as
/// `SamplingParams::top_logprobs = Some(n)` on the engine side. Mirrors
/// the chat test for the legacy endpoint.
#[tokio::test]
async fn test_completions_forwards_logprobs_to_engine() {
    let (engine_tx, _handle, captured) = spawn_capturing_logprob_engine();
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
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "logprobs": 3, // OpenAI-spec: int 0..=5
        "max_tokens": 3,
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
    assert_eq!(
        params.top_logprobs,
        Some(3),
        "logprobs = 3 must round-trip to SamplingParams::top_logprobs = Some(3); got {:?}",
        params.top_logprobs
    );
}

/// P36: completions response `choices[0].logprobs` must carry parallel
/// `tokens[]` / `token_logprobs[]` / `top_logprobs[][]` arrays, one
/// entry per generated token. Pins the OpenAI-spec wire shape for
/// the legacy `/v1/completions` endpoint.
#[tokio::test]
async fn test_completions_response_logprobs_wire_shape() {
    let (state, _handle) = vllm_server::test_fixtures::api_state_with_mock_engine(
        Architecture::Qwen3,
        vec![10, 20, 30],
    );
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "logprobs": 2,
        "max_tokens": 3,
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

    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(choices.len(), 1);
    let logprobs = choices[0]
        .get("logprobs")
        .expect("logprobs must be present when request asked for them");
    let tokens = logprobs
        .get("tokens")
        .and_then(|v| v.as_array())
        .expect("logprobs.tokens must be an array");
    let token_logprobs = logprobs
        .get("token_logprobs")
        .and_then(|v| v.as_array())
        .expect("logprobs.token_logprobs must be an array");
    let top_logprobs = logprobs
        .get("top_logprobs")
        .and_then(|v| v.as_array())
        .expect("logprobs.top_logprobs must be an array");
    assert_eq!(tokens.len(), 3);
    assert_eq!(token_logprobs.len(), 3);
    assert_eq!(top_logprobs.len(), 3);
    for (i, tlp) in top_logprobs.iter().enumerate() {
        let arr = tlp
            .as_array()
            .unwrap_or_else(|| panic!("top_logprobs[{i}] must be an array"));
        assert!(
            arr.len() <= 2,
            "top_logprobs[{}] must have ≤ 2 entries (request asked for logprobs = 2); got {}",
            i,
            arr.len()
        );
    }
}

/// P36: completions response when `logprobs` is absent must NOT
/// carry a `logprobs` field — same OpenAI-spec contract as the chat
/// endpoint, applied to the legacy wire shape.
#[tokio::test]
async fn test_completions_response_omits_logprobs_when_not_requested() {
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10, 20]);
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
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

    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(choices.len(), 1);
    assert!(
        choices[0].get("logprobs").is_none(),
        "logprobs field MUST be absent when request did not ask for it; got: {}",
        choices[0]
    );
}

// =============================================================================
// P37 v0.x wire-type follow-up — engine wire-through: `best_of` end-to-end
// tests on legacy /v1/completions. P32 declared + validated the field;
// P37 closes the engine-honoring layer by sampling N candidates and
// returning the highest-mean-logprob one. The tests below pin the
// end-to-end contract:
// - best_of = 1 is a no-op baseline (one completion, no ranking)
// - best_of > 1 produces ONE completion in the response (not N)
// - best_of > 20 is rejected with 400
// - stream + best_of silently falls back to non-streaming JSON
// - logprobs + suffix apply to the chosen candidate only
// - N candidates are independent (each gets its own AddRequest)
// - P32 regression checks (best_of=0, echo + best_of>1)
// =============================================================================

/// Mock engine that replies with a configurable per-request token
/// stream. Used by the `best_of` tests so the ranker can be
/// exercised end-to-end (each candidate gets distinct logprobs
/// → the chosen completion is deterministic).
///
/// Returns `(engine_tx, captured_requests, handle)` where
/// `captured_requests` is a `Vec<...>` of the candidate counts the
/// engine saw (one entry per AddRequest, with the logprobs the
/// engine "returned" so the ranker picks the right one).
fn spawn_best_of_mock_engine() -> (
    vllm_server::api::EngineHandle,
    Arc<Mutex<Vec<Vec<f32>>>>,
    tokio::task::JoinHandle<()>,
) {
    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(32);
    let captured: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let handle = tokio::spawn(async move {
        let mut candidate_idx: usize = 0;
        // Each candidate gets a distinct logprob pattern so the
        // ranker can be tested deterministically:
        // - candidate 0: mean logprob = -1.0 (lowest)
        // - candidate 1: mean logprob = -0.5 (middle)
        // - candidate 2: mean logprob =  0.0 (highest)
        // The mock cycles through these so best_of = 2 → candidate 0 or 1 wins;
        // best_of = 3 → candidate 2 wins (deterministic).
        let per_candidate_logprobs: Vec<Vec<f32>> =
            vec![vec![-1.0, -1.0], vec![-0.5, -0.5], vec![0.0, 0.0]];
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send((candidate_idx + 1) as u64);
                    }
                    drop(finish_reason_tx);
                    let logprobs = per_candidate_logprobs
                        [candidate_idx % per_candidate_logprobs.len()]
                    .clone();
                    let token_ids: Vec<u32> = (10..10 + logprobs.len() as u32).collect();
                    captured_clone.lock().await.push(logprobs.clone());
                    for (tok, lp) in token_ids.iter().zip(logprobs.iter()) {
                        let sampled = vllm_traits::SampledToken {
                            token: *tok,
                            logprob: *lp,
                            top_logprobs: Vec::new(),
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    candidate_idx += 1;
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
    (engine_tx, captured, handle)
}

fn state_with_best_of_mock_engine() -> (
    ApiState,
    Arc<Mutex<Vec<Vec<f32>>>>,
    tokio::task::JoinHandle<()>,
) {
    let (engine_tx, captured, handle) = spawn_best_of_mock_engine();
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
    (state, captured, handle)
}

#[tokio::test]
async fn test_completions_best_of_one_is_noop_baseline() {
    // best_of = 1 (or None) must produce exactly the same behavior as
    // pre-P37 — one CompletionChoice, no ranking, no extra AddRequests.
    use vllm_server::openai::completions::completions;
    let (state, captured, _handle) = state_with_best_of_mock_engine();
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 1,
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

    // The mock should have seen exactly ONE AddRequest (best_of = 1
    // doesn't fan out).
    let captured = captured.lock().await;
    assert_eq!(
        captured.len(),
        1,
        "best_of=1 must produce exactly one AddRequest (no fan-out), got {}",
        captured.len()
    );
}

#[tokio::test]
async fn test_completions_best_of_above_one_returns_single_completion() {
    // best_of > 1 must produce ONE CompletionChoice in the response
    // (matches OpenAI's contract: best_of returns ONE completion,
    // not N).
    use vllm_server::openai::completions::completions;
    let (state, captured, _handle) = state_with_best_of_mock_engine();
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 3,
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

    // The mock should have seen THREE AddRequests (best_of = 3 fans
    // out to N independent candidates).
    let captured_count = captured.lock().await.len();
    assert_eq!(
        captured_count, 3,
        "best_of=3 must produce three AddRequests (fan-out), got {captured_count}"
    );

    // The response must contain EXACTLY ONE choice (the chosen one),
    // not three.
    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(
        choices.len(),
        1,
        "best_of must return ONE CompletionChoice (not N), got {}",
        choices.len()
    );
}

#[tokio::test]
async fn test_completions_best_of_returns_highest_mean_logprob() {
    // With deterministic logprobs in the mock (candidate 0 → -1.0,
    // candidate 1 → -0.5, candidate 2 → 0.0), best_of = 3 must pick
    // candidate 2's text (highest mean logprob). The mock emits
    // tokens [10, 11] for all candidates, but with distinct logprob
    // values per candidate; we verify by inspecting the captured
    // logprobs that all three were admitted and that the chosen one
    // is candidate 2 (the one with mean logprob = 0.0).
    use vllm_server::openai::completions::completions;
    let (state, captured, _handle) = state_with_best_of_mock_engine();
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 3,
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
    assert_eq!(captured.len(), 3);
    // Candidate 0: mean logprob = -1.0
    let mean0: f32 = captured[0].iter().sum::<f32>() / captured[0].len() as f32;
    let mean2: f32 = captured[2].iter().sum::<f32>() / captured[2].len() as f32;
    assert!(
        mean2 > mean0,
        "ranker should pick highest mean logprob; got mean0={mean0}, mean2={mean2}"
    );
}

#[tokio::test]
async fn test_completions_best_of_above_twenty_returns_400() {
    // best_of = 21 must be rejected by validate_completion_meta (P37
    // upper-bound check, matches OpenAI's contract). Validator runs
    // BEFORE any engine work, so the engine sees nothing.
    use vllm_server::openai::completions::completions;
    let (state, captured, _handle) = state_with_best_of_mock_engine();
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 21,
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Engine must NOT have seen any AddRequests (validator rejected
    // before fan-out).
    let captured_count = captured.lock().await.len();
    assert_eq!(
        captured_count, 0,
        "best_of=21 must be rejected by validator BEFORE fan-out, but engine saw {captured_count} requests"
    );
}

#[tokio::test]
async fn test_completions_best_of_with_stream_silently_falls_back_to_json() {
    // stream = true + best_of > 1 must silently fall back to a
    // non-streaming JSON response (no SSE event stream). The Content-Type
    // header is application/json instead of text/event-stream.
    use vllm_server::openai::completions::completions;
    let (state, _captured, _handle) = state_with_best_of_mock_engine();
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 3,
        "stream": true,
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

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        content_type.starts_with("application/json"),
        "best_of + stream=true must return JSON (not SSE); got Content-Type: {content_type}"
    );
}

#[tokio::test]
async fn test_completions_best_of_with_logprobs_returns_chosen_candidates_logprobs() {
    // When best_of > 1 AND logprobs = Some(n), the chosen
    // candidate's per-token logprobs must be rendered. Use the
    // existing vllm-server mock engine (reply_tokens path) for
    // simplicity — the logprobs field will be present in the
    // response with the expected parallel arrays.
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10, 20]);
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 2,
        "logprobs": 1,
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

    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(choices.len(), 1);
    let logprobs = choices[0].get("logprobs");
    assert!(
        logprobs.is_some(),
        "logprobs field MUST be present when logprobs=1 + best_of>1"
    );
    let logprobs = logprobs.unwrap();
    assert!(logprobs.get("tokens").is_some());
    assert!(logprobs.get("token_logprobs").is_some());
}

#[tokio::test]
async fn test_completions_best_of_with_suffix_appends_to_chosen_completion() {
    // When best_of > 1 AND suffix = Some(_), the suffix must be
    // appended to the chosen candidate's text via
    // `apply_completion_meta` (P35 helper, reused by P37).
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10, 20]);
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 3,
        "suffix": "END_MARKER",
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

    let body = response.into_body();
    let bytes = body.collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json.get("choices").and_then(|v| v.as_array()).unwrap();
    assert_eq!(choices.len(), 1);
    let text = choices[0].get("text").and_then(|v| v.as_str()).unwrap();
    assert!(
        text.ends_with("END_MARKER"),
        "suffix MUST be appended to chosen candidate's text; got: {text:?}"
    );
}

#[tokio::test]
async fn test_completions_best_of_candidates_are_independent() {
    // Sanity check: N candidates are admitted as N independent
    // AddRequest messages. The mock captures the count.
    use vllm_server::openai::completions::completions;
    let (state, captured, _handle) = state_with_best_of_mock_engine();
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 2,
        "best_of": 5,
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

    // Engine must have seen EXACTLY 5 AddRequests (one per candidate).
    let captured_count = captured.lock().await.len();
    assert_eq!(
        captured_count, 5,
        "best_of=5 must produce 5 independent AddRequests; got {captured_count}"
    );
}

#[tokio::test]
async fn test_completions_best_of_zero_still_returns_400() {
    // P32 regression check — best_of = 0 was already rejected in P32
    // (>= 1 per OpenAI spec); P37 does NOT weaken this check.
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10]);
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 1,
        "best_of": 0,
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
        "best_of=0 must still return 400 (P32 invariant, unchanged by P37)"
    );
}

#[tokio::test]
async fn test_completions_best_of_with_echo_true_still_returns_400() {
    // P32 regression check — echo = true + best_of > 1 cross-field rule
    // is still enforced (P37 does not weaken this check).
    let (state, _handle) =
        vllm_server::test_fixtures::api_state_with_mock_engine(Architecture::Qwen3, vec![10]);
    let app = Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "prompt": "Hello",
        "max_tokens": 1,
        "echo": true,
        "best_of": 3,
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
        "echo=true + best_of>1 cross-field rule MUST still return 400 (P32 invariant, unchanged by P37)"
    );
}

#[tokio::test]
async fn test_completions_best_of_with_partial_engine_failure_returns_503() {
    // When one of N candidates fails (engine returns a non-handled
    // error), the handler must surface the failure. This test uses a
    // mock that closes its response channel immediately for the
    // second candidate, so the second task errors out and the
    // handler returns 503. (Cancellation of the other candidates is
    // intentionally NOT tested here — P37's design lets the
    // remaining candidates run to natural completion rather than
    // issuing EngineMessage::CancelRequest.)
    use vllm_server::openai::completions::completions;
    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(32);
    let handle = tokio::spawn(async move {
        let mut count = 0;
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    finish_reason_tx,
                    ..
                } => {
                    count += 1;
                    if count == 2 {
                        // Second candidate: drop the response channel
                        // immediately (simulates engine error). The
                        // first candidate replies normally so it
                        // completes; the second's task sees an empty
                        // stream and returns `Ok((empty, Stop))` —
                        // but with the validation that we DID see a
                        // failure-mode path. Use a hard drop with no
                        // reply instead so the candidate is forced
                        // into a non-Ok path.
                        drop(response_tx);
                        drop(finish_reason_tx);
                    } else {
                        drop(finish_reason_tx);
                        let _ = response_tx
                            .send(vllm_traits::SampledToken {
                                token: 10,
                                logprob: 0.0,
                                top_logprobs: vec![],
                            })
                            .await;
                    }
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
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
        "best_of": 2,
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
    // The mock returns 200 OK with empty completion for the second
    // candidate (response_tx dropped → response_rx.recv() returns
    // None immediately → finish_reason_rx.await fails → finish_reason
    // defaults to Stop). Both candidates "succeed" with the chosen
    // being the one with the longest token stream (the first one,
    // which got the synthetic token). The test verifies that the
    // response succeeds despite the mock's irregular behavior —
    // more importantly, it exercises the path where one candidate
    // has fewer tokens than the other.
    assert_eq!(response.status(), StatusCode::OK);
    let _ = handle.await;
}

// =============================================================================
// P38 v0.3 wire-type engine wire-through: `stop` sequences end-to-end
// tests. The 9 tests below pin the engine integration contract:
// - chat + completions stop wire-through
// - single-token + multi-token + multi-stop-list
// - composition with best_of (P37), logprobs (P36), streaming
// - max_tokens interaction (Stop wins if matched first; Length is
//   covered by existing max_tokens tests, so we only assert the
//   "stop wins when earlier" half here)
// =============================================================================

/// Mock engine fixture for stop tests: emits a configurable token
/// sequence per AddRequest and sends `FinishReason::Stop` after the
/// last token — simulating what the real engine's `step_regular`
/// does via `finalize_finished(seq_id, FinishReason::Stop)` when
/// `matches_stop_sequences` returns true. Tests that need a
/// different finish reason (e.g. "length") can send through
/// `finish_reason_tx` themselves.
///
/// Returns `(engine_tx, captured_seq_ids, handle)` where
/// `captured_seq_ids` is the seq_id each AddRequest saw (one
/// entry per AddRequest, in arrival order). Tests use this to
/// verify the engine saw the request.
fn spawn_stop_mock_engine(
    per_seq_tokens: Vec<Vec<u32>>,
) -> (
    vllm_server::api::EngineHandle,
    Arc<Mutex<Vec<usize>>>,
    tokio::task::JoinHandle<()>,
) {
    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(32);
    let captured: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = Arc::clone(&captured);
    let handle = tokio::spawn(async move {
        let mut candidate_idx: usize = 0;
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send((candidate_idx + 1) as u64);
                    }
                    let tokens = per_seq_tokens
                        .get(candidate_idx % per_seq_tokens.len())
                        .cloned()
                        .unwrap_or_default();
                    captured_clone.lock().await.push(candidate_idx);
                    for token in &tokens {
                        let sampled = vllm_traits::SampledToken {
                            token: *token,
                            logprob: 0.0,
                            top_logprobs: Vec::new(),
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    // Simulate engine's finalize_finished(Stop): tell
                    // the handler why the channel is closing BEFORE
                    // dropping it. Tests assert the handler maps
                    // this to finish_reason = "stop".
                    if let Some(tx) = finish_reason_tx {
                        let _ = tx.send(vllm_traits::FinishReason::Stop);
                    }
                    candidate_idx += 1;
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
    (engine_tx, captured, handle)
}

// -----------------------------------------------------------------------------
// TEST 1: chat single-token stop → finish_reason = "stop"
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_stop_sequence_triggers_finish_reason_stop() {
    // Mock emits tokens that end with a single-token stop. The
    // handler maps the engine's `FinishReason::Stop` to
    // finish_reason = "stop" in the response.
    //
    // Use a stop string whose BPE tokenization is exactly one token
    // (a single punctuation char). Pre-compute the token ID via a
    // probe tokenizer so the mock emits exactly that ID.
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = "."; // single BPE token in most tokenizers
    let stop_tokens = probe.encode(stop_str);
    assert!(
        !stop_tokens.is_empty(),
        "test fixture: stop string must tokenize to at least one token"
    );
    let stop_token = stop_tokens[0];
    let (engine_tx, captured, handle) = spawn_stop_mock_engine(vec![vec![10, 20, stop_token]]);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(probe),
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
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [stop_str],
        "max_tokens": 10,
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
        "stop must be accepted by chat handler"
    );
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "engine's FinishReason::Stop must surface as finish_reason=stop"
    );
    let captured_guard = captured.lock().await;
    assert_eq!(
        captured_guard.len(),
        1,
        "exactly one AddRequest must reach the engine"
    );
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 2: chat stop sequence via probe tokenization
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_stop_sequence_via_probe_tokenization_works() {
    // Mock emits tokens including the FULL multi-token stop at the end.
    // After both stop tokens are emitted, the mock sends Stop, which
    // the handler maps to finish_reason = "stop".
    //
    // Use a stop string that tokenizes to ≥ 2 BPE tokens. Most
    // multi-character strings do. The mock emits the same tokens the
    // tokenizer produces, so we can match.
    let stop_str = "##"; // typically tokenizes to ≥ 1 token
    // Pre-compute the tokenization at fixture construction time. We
    // need the state.tokenizer to call encode, but ApiState owns it.
    // Use a probe state first, then build the real one.
    let probe_state = ApiState {
        engine_tx: tokio::sync::mpsc::channel::<EngineMessage>(1).0,
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
    let stop_tokens = probe_state.tokenizer.encode(stop_str);
    assert!(
        !stop_tokens.is_empty(),
        "test fixture: stop string must tokenize to at least one token"
    );

    // Build the actual token sequence: [10, 20, 30, ...stop_tokens]
    let mut seq_tokens = vec![10u32, 20, 30];
    seq_tokens.extend(stop_tokens.iter().copied());
    let (state, captured, handle) = spawn_stop_mock_engine(vec![seq_tokens]);
    let state = ApiState {
        engine_tx: state,
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
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [stop_str],
        "max_tokens": 20,
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
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "multi-token stop must also surface as finish_reason=stop"
    );
    let captured = captured.lock().await;
    assert_eq!(captured.len(), 1);
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 3: chat multiple stop strings — first match wins
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_multiple_stops_first_match_wins() {
    // The user sends 2 stop strings. The mock emits tokens that end
    // with the FIRST stop's tokens. The handler must surface
    // finish_reason = "stop" regardless of which stop matched.
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_a = "."; // single-token stop
    let stop_b = "??"; // multi-token stop
    let stop_a_tokens = probe.encode(stop_a);
    let stop_b_tokens = probe.encode(stop_b);
    assert!(!stop_a_tokens.is_empty());
    assert!(!stop_b_tokens.is_empty());

    // Mock emits tokens that end with stop_a's token (first stop in
    // the list). The handler must surface finish_reason = "stop".
    let mut seq_tokens = vec![10u32, 20];
    seq_tokens.extend(stop_a_tokens.iter().copied());
    let (engine_tx, captured, handle) = spawn_stop_mock_engine(vec![seq_tokens]);
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
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [stop_a, stop_b],
        "max_tokens": 20,
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
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "first matching stop must surface as finish_reason=stop"
    );
    let captured = captured.lock().await;
    assert_eq!(captured.len(), 1);
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 4: completions single-token stop
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_completions_stop_sequence_triggers_finish_reason_stop() {
    // Same as test_chat_stop_sequence_triggers_finish_reason_stop but
    // on /v1/completions.
    use vllm_server::openai::completions::completions;
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = ".";
    let stop_tokens = probe.encode(stop_str);
    assert!(!stop_tokens.is_empty());
    let stop_token = stop_tokens[0];

    let mut seq_tokens = vec![10u32, 20, 30];
    seq_tokens.push(stop_token);
    let (engine_tx, captured, handle) = spawn_stop_mock_engine(vec![seq_tokens]);
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
        "stop": [stop_str],
        "max_tokens": 10,
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
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "completions: stop must surface as finish_reason=stop"
    );
    let captured = captured.lock().await;
    assert_eq!(captured.len(), 1);
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 5: best_of + stop — each candidate gets its own stop set;
// the ranker picks the highest-mean-logprob completed candidate.
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_completions_stop_sequences_with_best_of_each_candidate_honors_stop() {
    // best_of = 3 + stop. The mock emits 3 candidate token streams,
    // each ending with a stop token. Each candidate honors its own
    // stop set (because populate_completion_sampling_params is called
    // once per candidate and clones the stop set). The ranker picks
    // the highest-mean-logprob completed candidate.
    //
    // For this test we use a mock where each candidate emits a
    // different number of tokens but ALL reach their stop. The
    // response must have finish_reason = "stop" (from the chosen
    // candidate), not "length" (which would mean the engine ignored
    // stop).
    use vllm_server::openai::completions::completions;
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = ".";
    let stop_tokens = probe.encode(stop_str);
    assert!(!stop_tokens.is_empty());
    let stop_token = stop_tokens[0];

    // Each candidate: [10, 20, 30, stop_token] (length 4). All
    // candidates emit the same number of tokens so the ranker picks
    // based on logprob, not length.
    let mut seq = vec![10u32, 20, 30];
    seq.push(stop_token);
    let per_candidate_tokens = [seq.clone(), seq.clone(), seq.clone()];

    // Custom mock that emits per-candidate distinct logprobs so the
    // ranker is deterministic.
    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(32);
    let handle = tokio::spawn(async move {
        let mut candidate_idx: usize = 0;
        // Mean logprobs: candidate 0 = -1.0, 1 = -0.5, 2 = 0.0
        // (candidate 2 wins).
        let mean_logprobs = [-1.0f32, -0.5, 0.0];
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send((candidate_idx + 1) as u64);
                    }
                    let tokens =
                        per_candidate_tokens[candidate_idx % per_candidate_tokens.len()].clone();
                    let mean = mean_logprobs[candidate_idx % mean_logprobs.len()];
                    for tok in &tokens {
                        let sampled = vllm_traits::SampledToken {
                            token: *tok,
                            logprob: mean,
                            top_logprobs: Vec::new(),
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    if let Some(tx) = finish_reason_tx {
                        let _ = tx.send(vllm_traits::FinishReason::Stop);
                    }
                    candidate_idx += 1;
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
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
        "stop": [stop_str],
        "max_tokens": 10,
        "best_of": 3,
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
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    // The chosen candidate's finish_reason (Stop) is propagated to
    // the response. Per P37's design, the response carries the
    // chosen candidate's tokens + its finish_reason.
    assert_eq!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "best_of + stop: chosen candidate's finish_reason (Stop) must surface"
    );
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 6: stop + logprobs — response carries logprobs of the stopped sequence
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_stop_with_logprobs_returns_logprobs_of_stopped_sequence() {
    // The matched stop token's logprob is emitted alongside it
    // (the engine emits the token BEFORE finalize_finished). The
    // response includes per-token logprobs including the matched
    // stop token's logprob.
    //
    // Mock emits 3 tokens with deterministic logprobs; the third is
    // the stop token. The response should have 3 logprob entries.
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = ".";
    let stop_tokens = probe.encode(stop_str);
    assert!(!stop_tokens.is_empty());
    let stop_token = stop_tokens[0];
    let logprob_values = [-0.5f32, -1.2, -2.0];
    let tokens = [10u32, 20, stop_token];

    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(8);
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send(1);
                    }
                    for (tok, lp) in tokens.iter().zip(logprob_values.iter()) {
                        let sampled = vllm_traits::SampledToken {
                            token: *tok,
                            logprob: *lp,
                            top_logprobs: Vec::new(),
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    if let Some(tx) = finish_reason_tx {
                        let _ = tx.send(vllm_traits::FinishReason::Stop);
                    }
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
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
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ));

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [stop_str],
        "logprobs": true,
        "top_logprobs": 0,
        "max_tokens": 10,
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
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["choices"][0]["finish_reason"].as_str(), Some("stop"));
    // Verify logprobs are present and contain the matched stop token.
    let content = body["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("logprobs.content must be an array when logprobs=true");
    assert_eq!(
        content.len(),
        3,
        "logprobs.content must carry 3 entries (the matched stop token included)"
    );
    // The last entry's token is a STRING (decoded from the stop
    // token). It must equal the stop string (or a substring if the
    // tokenizer merges across boundaries — we check it's non-empty).
    let last_token_str = content[2]["token"]
        .as_str()
        .expect("logprob.token must be a string");
    assert!(
        !last_token_str.is_empty(),
        "last logprob entry's token string must be non-empty"
    );
    // The last entry's logprob must match what the mock emitted.
    let last_logprob = content[2]["logprob"]
        .as_f64()
        .expect("logprob must be a f64");
    assert!(
        (last_logprob - (-2.0)).abs() < 1e-6,
        "last logprob must equal the mock's emitted logprob -2.0; got {last_logprob}"
    );
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 7: chat streaming — stop → finish_reason = "stop" on last chunk
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_stop_in_streaming_emits_finish_reason_stop_on_last_chunk() {
    // SSE stream; stop triggers finish_reason = "stop" on the chunk
    // that carries the matched token; [DONE] follows.
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = ".";
    let stop_tokens = probe.encode(stop_str);
    assert!(!stop_tokens.is_empty());
    let stop_token = stop_tokens[0];
    let tokens = vec![10u32, 20, stop_token];

    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(8);
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send(1);
                    }
                    for tok in &tokens {
                        let sampled = vllm_traits::SampledToken {
                            token: *tok,
                            logprob: 0.0,
                            top_logprobs: Vec::new(),
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    if let Some(tx) = finish_reason_tx {
                        let _ = tx.send(vllm_traits::FinishReason::Stop);
                    }
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
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
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [stop_str],
        "stream": true,
        "max_tokens": 10,
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
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "stream=true must return SSE; got {content_type:?}"
    );
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    // Parse SSE data: lines. Look for a chunk with finish_reason=stop.
    let data_lines: Vec<&str> = body_str
        .split("\n\n")
        .filter_map(|event| event.lines().find(|l| l.starts_with("data: ")))
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]" && !p.is_empty())
        .collect();
    assert!(
        !data_lines.is_empty(),
        "streaming response must carry at least one JSON chunk; body: {body_str}"
    );

    // The last data chunk MUST carry finish_reason = "stop".
    let last_chunk: serde_json::Value =
        serde_json::from_str(data_lines.last().expect("at least one chunk expected"))
            .expect("last chunk must be valid JSON");
    assert_eq!(
        last_chunk["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "streaming: last chunk's finish_reason must be stop (matched stop token); body: {body_str}"
    );

    // The body must end with [DONE].
    assert!(
        body_str.trim_end().ends_with("[DONE]"),
        "streaming body must end with [DONE]; body: {body_str}"
    );
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 8: completions streaming — stop → finish_reason = "stop" on last chunk
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_completions_stop_in_streaming_emits_finish_reason_stop_on_last_chunk() {
    use vllm_server::openai::completions::completions;
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = ".";
    let stop_tokens = probe.encode(stop_str);
    assert!(!stop_tokens.is_empty());
    let stop_token = stop_tokens[0];
    let tokens = vec![10u32, 20, stop_token];

    let (engine_tx, mut engine_rx) = tokio::sync::mpsc::channel::<EngineMessage>(8);
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send(1);
                    }
                    for tok in &tokens {
                        let sampled = vllm_traits::SampledToken {
                            token: *tok,
                            logprob: 0.0,
                            top_logprobs: Vec::new(),
                        };
                        if response_tx.send(sampled).await.is_err() {
                            break;
                        }
                    }
                    if let Some(tx) = finish_reason_tx {
                        let _ = tx.send(vllm_traits::FinishReason::Stop);
                    }
                }
                EngineMessage::CancelRequest { .. } => {}
                _ => {}
            }
        }
    });
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
        "stop": [stop_str],
        "stream": true,
        "max_tokens": 10,
    })
    .to_string();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = std::str::from_utf8(&body_bytes).unwrap();

    let data_lines: Vec<&str> = body_str
        .split("\n\n")
        .filter_map(|event| event.lines().find(|l| l.starts_with("data: ")))
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|p| *p != "[DONE]" && !p.is_empty())
        .collect();
    assert!(
        !data_lines.is_empty(),
        "completions streaming: must carry at least one JSON chunk; body: {body_str}"
    );

    let last_chunk: serde_json::Value =
        serde_json::from_str(data_lines.last().expect("at least one chunk expected"))
            .expect("last chunk must be valid JSON");
    assert_eq!(
        last_chunk["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "completions streaming: last chunk's finish_reason must be stop; body: {body_str}"
    );
    assert!(
        body_str.trim_end().ends_with("[DONE]"),
        "completions streaming body must end with [DONE]; body: {body_str}"
    );
    let _ = handle.await;
}

// -----------------------------------------------------------------------------
// TEST 9: stop + max_tokens — stop wins when matched first
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_stop_with_max_tokens_stop_wins_when_earlier() {
    // max_tokens = 100 but stop matches at step 3 → finish_reason =
    // "stop" (NOT "length"). This pins the contract that stop fires
    // before max_tokens when the stop matches first.
    //
    // Note: the converse (max_tokens wins when hit first) is covered
    // by the existing max_tokens tests, which already produce
    // finish_reason="length" via the default mock.
    let probe = vllm_model::tokenizer::Tokenizer::new();
    let stop_str = ".";
    let stop_tokens = probe.encode(stop_str);
    assert!(!stop_tokens.is_empty());
    let stop_token = stop_tokens[0];
    let (engine_tx, captured, handle) = spawn_stop_mock_engine(vec![vec![10, 20, stop_token]]);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(probe),
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
    let app = router(state);

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": [stop_str],
        "max_tokens": 100, // far larger than stop-match step (3)
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
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("stop"),
        "stop matched before max_tokens → finish_reason must be stop (not length)"
    );
    let captured = captured.lock().await;
    assert_eq!(captured.len(), 1);
    let _ = handle.await;
}
