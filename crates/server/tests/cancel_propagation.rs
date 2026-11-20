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

/// Router exposing `/v1/completions` (the legacy endpoint) with the same
/// correlation middleware as production.
fn completions_router(state: ApiState) -> Router {
    Router::new()
        .route(
            "/v1/completions",
            post(vllm_server::openai::completions::completions),
        )
        .with_state(state)
        .layer(axum::middleware::from_fn(
            vllm_server::security::correlation::correlation_id_middleware,
        ))
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
    spawn_recording_mock_engine_inner(tokens, false)
}

/// Same engine, but it parks after the last token instead of closing the
/// response channel — a generation that is still running when the client
/// goes away.
fn spawn_stalled_mock_engine(
    tokens: Vec<TokenId>,
) -> (EngineHandle, Arc<tokio::sync::Mutex<Vec<u64>>>) {
    spawn_recording_mock_engine_inner(tokens, true)
}

fn spawn_recording_mock_engine_inner(
    tokens: Vec<TokenId>,
    stall: bool,
) -> (EngineHandle, Arc<tokio::sync::Mutex<Vec<u64>>>) {
    let cancels: Arc<tokio::sync::Mutex<Vec<u64>>> = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let cancels_clone = cancels.clone();

    let (engine_tx, mut engine_rx) = mpsc::channel(TEST_MAILBOX);
    tokio::spawn(async move {
        let mut next_seq_id: u64 = 1;
        // Senders held open by the `stall` mode, so their channel never closes.
        let mut held: Vec<tokio::sync::mpsc::Sender<vllm_traits::SampledToken>> = Vec::new();
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
                    if stall {
                        // Keep the sequence alive (the handler waits on a
                        // channel that never closes) WITHOUT parking the
                        // engine loop — it still has to receive the
                        // `CancelRequest` this test is about.
                        held.push(response_tx);
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

/// A non-streaming client that gives up must also release the sequence.
///
/// The streaming paths install a `CancelOnDrop` guard because an SSE body can
/// be dropped mid-stream. The non-streaming handler sends
/// `seq_id_tx: None` and awaits the token loop to completion, so when axum
/// drops the handler future (client closed the connection while waiting for
/// JSON) nothing tells the engine to stop: it keeps generating up to
/// `max_tokens`, holding a scheduler slot and KV blocks for a caller that is
/// gone. `max_tokens` is caller-chosen, so this is unbounded work.
#[tokio::test]
async fn non_streaming_client_disconnect_sends_cancel_request() {
    let (engine_tx, cancels) = spawn_stalled_mock_engine(Vec::new());
    let state = build_state(engine_tx);
    let app = router(state);

    let body = serde_json::json!({
        "model": "llama-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "max_tokens": 4096
    })
    .to_string();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    // Poll the handler once so it enqueues the AddRequest and parks on the
    // token channel, then drop the future — what axum does when the client
    // disconnects before the JSON reply is ready.
    // Spawn the handler and abort it while it waits for engine output —
    // the same shape as axum dropping the handler future when the client
    // closes the connection before the JSON reply is ready.
    let handle = tokio::spawn(async move { app.oneshot(request).await });
    // Let the handler enqueue the AddRequest and complete the seq_id
    // handshake with the mock engine.
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!handle.is_finished(), "the handler must still be waiting");
    handle.abort();
    let _ = handle.await;

    // Give the drop path a moment to enqueue the cancel.
    for _ in 0..50 {
        if !cancels.lock().await.is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let recorded = cancels.lock().await.clone();
    assert_eq!(
        recorded,
        vec![1],
        "dropping a non-streaming handler must cancel its sequence; got {recorded:?}"
    );
}

/// A non-streaming `n > 1` request that the client abandons must release
/// every candidate sequence.
///
/// The `n > 1` fan-out spawns one tokio task per candidate and each sends
/// `seq_id_tx: None`, so an abandoned request leaves N sequences generating
/// to `max_tokens` — N times the wasted work of the single-candidate case
/// fixed alongside this test.
#[tokio::test]
async fn non_streaming_n_gt_1_disconnect_cancels_every_candidate() {
    let (engine_tx, cancels) = spawn_stalled_mock_engine(Vec::new());
    let state = build_state(engine_tx);
    let app = router(state);

    let body = serde_json::json!({
        "model": "llama-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false,
        "n": 3,
        "max_tokens": 4096
    })
    .to_string();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let handle = tokio::spawn(async move { app.oneshot(request).await });
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    assert!(!handle.is_finished(), "the handler must still be waiting");
    handle.abort();
    let _ = handle.await;

    let mut recorded = Vec::new();
    for _ in 0..50 {
        recorded = cancels.lock().await.clone();
        if recorded.len() >= 3 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    recorded.sort_unstable();
    assert_eq!(
        recorded,
        vec![1, 2, 3],
        "all three candidate sequences must be cancelled, got {recorded:?}"
    );
}

/// The legacy `/v1/completions` candidate path needs the same guard: it had
/// the identical `seq_id_tx: None` shape as the chat candidate, so an
/// abandoned `n > 1` completions request also left N sequences generating.
#[tokio::test]
async fn non_streaming_completions_disconnect_sends_cancel_request() {
    let (engine_tx, cancels) = spawn_stalled_mock_engine(Vec::new());
    let state = build_state(engine_tx);
    let app = completions_router(state);

    let body = serde_json::json!({
        "model": "llama-test",
        "prompt": "Hello",
        "stream": false,
        "n": 2,
        "max_tokens": 4096
    })
    .to_string();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let handle = tokio::spawn(async move { app.oneshot(request).await });
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    assert!(!handle.is_finished(), "the handler must still be waiting");
    handle.abort();
    let _ = handle.await;

    let mut recorded = Vec::new();
    for _ in 0..50 {
        recorded = cancels.lock().await.clone();
        if recorded.len() >= 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    recorded.sort_unstable();
    assert_eq!(
        recorded,
        vec![1, 2],
        "both completions candidates must be cancelled, got {recorded:?}"
    );
}
