//! Streaming cancel propagation test.
//!
//! Production-readiness recommendation: when an SSE client
//! disconnects mid-stream, the HTTP handler must release the
//! engine-side resources (KV blocks, sequence slot, response
//! channel) so the engine stops generating tokens for a caller
//! that has already gone away. Without this, a client that closes
//! its TCP connection after the first chunk still occupies a
//! scheduler slot until `max_tokens` — wasting GPU cycles and
//! blocking other requests.
//!
//! This test wires a recording mock engine that captures every
//! `EngineMessage::CancelRequest` it sees, sends a streaming
//! `/v1/chat/completions` request, then drops the response body
//! BEFORE consuming the second token. The handler's
//! `CancelOnDrop` guard fires, the mock engine receives a
//! `CancelRequest { seq_id }` with the same `seq_id` it assigned
//! in `AddRequest`, and we assert that's the only cancel that
//! arrived (no redundant cancels for naturally-completed streams).

#![cfg(test)]

use std::sync::Arc;

use axum::{Router, body::Body, http::Request, routing::post};
use http_body_util::BodyExt;
use tokio::sync::mpsc;
use tower::ServiceExt;
use vllm_core::types::EngineMessage;
use vllm_model::config::Architecture;
use vllm_server::ApiState;
use vllm_server::api::EngineHandle;
use vllm_server::openai::batch::BatchManager;
use vllm_server::openai::chat::chat_completions;
use vllm_traits::TokenId;

const TEST_MAILBOX: usize = 16;

fn chat_request_json(model: &str) -> String {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "max_tokens": 50
    })
    .to_string()
}

fn build_state(engine_tx: EngineHandle) -> ApiState {
    ApiState {
        engine_tx,
        tokenizer: Arc::new(vllm_model::tokenizer::Tokenizer::new()),
        architecture: Architecture::Llama,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        audit: Arc::new(vllm_server::security::audit::AuditLogger::new(1000)),
        health: Arc::new(std::sync::RwLock::new(vllm_server::HealthChecker::new(
            true, true,
        ))),
        metrics: Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new()),
        max_model_len: None,
        arch_capabilities: None,
    }
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        // chat_completions now requires `Extension<CorrelationId>`
        // (P10 / production-readiness §6). Mount the same middleware
        // the production router uses so the tests exercise the real
        // boundary instead of returning 500 from a failed extractor.
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ))
}

/// Spawn a mock engine that records every `CancelRequest`
/// `seq_id` it sees and replies to `AddRequest` with a synthetic
/// `seq_id` + a stream of tokens. Returns the engine handle and a
/// shared vec of recorded cancel `seq_ids`.
fn spawn_recording_mock_engine(
    tokens: Vec<TokenId>,
) -> (EngineHandle, Arc<tokio::sync::Mutex<Vec<u64>>>) {
    let cancels: Arc<tokio::sync::Mutex<Vec<u64>>> = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let cancels_clone = cancels.clone();

    let (engine_tx, mut engine_rx) = mpsc::channel(TEST_MAILBOX);
    tokio::spawn(async move {
        let mut next_seq_id: u64 = 1;
        while let Some(msg) = engine_rx.recv().await {
            match msg {
                EngineMessage::AddRequest {
                    response_tx,
                    seq_id_tx,
                    finish_reason_tx,
                    ..
                } => {
                    let seq_id = next_seq_id;
                    next_seq_id += 1;
                    if let Some(tx) = seq_id_tx {
                        let _ = tx.send(seq_id);
                    }
                    // Mock engine doesn't simulate finalization —
                    // drop the reason so the handler falls back to
                    // `"stop"`.
                    drop(finish_reason_tx);
                    for token in &tokens {
                        if response_tx
                            .send(vllm_traits::SampledToken {
                                token: *token,
                                logprob: 0.0,
                                top_logprobs: vec![],
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
                EngineMessage::CancelRequest { seq_id } => {
                    cancels_clone.lock().await.push(seq_id);
                }
                _ => {}
            }
        }
    });
    (engine_tx, cancels)
}

#[tokio::test]
async fn streaming_client_disconnect_sends_cancel_request() {
    // 100 tokens: plenty for the body to still be streaming
    // when we drop it. The mock emits them as fast as the
    // channel allows; we drop the body after reading the
    // first SSE frame.
    let (engine_tx, cancels) =
        spawn_recording_mock_engine((0..100u32).map(|i| i as TokenId).collect());

    let state = build_state(engine_tx);
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json("llama-test")))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        axum::http::StatusCode::OK,
        "streaming endpoint must accept the request"
    );
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .contains("text/event-stream"),
        "response must be SSE"
    );

    // Pull the response body and read only the first frame,
    // then drop the body to simulate a client that closed its
    // TCP connection after seeing one event. We use `frame()`
    // (not `collect()`) so we definitely see at least one SSE
    // event arrive before the body is dropped.
    let mut body = response.into_body();
    let _first_frame = body
        .frame()
        .await
        .expect("first SSE frame must arrive")
        .expect("first SSE frame must not error");
    drop(body);

    // Yield many times so the Drop guard's `try_send` reaches
    // the mock engine and the spawned task processes it.
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }

    let recorded = cancels.lock().await.clone();
    assert_eq!(
        recorded.len(),
        1,
        "expected exactly one CancelRequest after client disconnect, got {recorded:?}"
    );
    let first = recorded[0];
    assert_ne!(
        first, 0,
        "cancel seq_id must be the engine-assigned id, not the rejection sentinel"
    );
}

#[tokio::test]
async fn streaming_natural_completion_does_not_cancel() {
    // Counterpart: a streaming client that consumes the full
    // body (natural completion) must NOT trigger a redundant
    // CancelRequest — the `disarm()` path is the only one that
    // exercises this invariant.
    let (engine_tx, cancels) = spawn_recording_mock_engine(vec![0, 1, 2]);

    let state = build_state(engine_tx);
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(chat_request_json("llama-test")))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    // Drain the body fully — this triggers the natural
    // completion path (`rx.recv()` returns None → guard.disarm()).
    // Use `collect` so the body finishes on its own without us
    // having to reason about SSE frame boundaries.
    let body = response.into_body();
    let _bytes = body
        .collect()
        .await
        .expect("body must finish without error");

    // Yield to let any in-flight messages settle.
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }

    let recorded = cancels.lock().await.clone();
    assert_eq!(
        recorded.len(),
        0,
        "natural completion must not send CancelRequest (disarmed); got {recorded:?}"
    );
}
