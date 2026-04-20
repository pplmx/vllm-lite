use axum::http::StatusCode;
use tokio::sync::mpsc;
use vllm_server::ApiState;
use vllm_server::openai::models::models_handler;

#[tokio::test]
async fn test_models_handler_returns_list() {
    let (tx, _rx) = mpsc::unbounded_channel();
    let tokenizer = Arc::new(vllm_model::tokenizer::Tokenizer::new());
    let batch_manager = Arc::new(vllm_server::openai::batch::manager::BatchManager::new());
    let health = Arc::new(std::sync::RwLock::new(
        vllm_server::health::HealthChecker::new(true, true),
    ));
    let metrics = Arc::new(vllm_core::metrics::EnhancedMetricsCollector::new());

    let state = ApiState {
        engine_tx: tx,
        tokenizer,
        batch_manager,
        auth: None,
        health,
        metrics,
    };

    let response = models_handler(axum::extract::State(state)).await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body();
    let bytes = axum::body::to_bytes(body, 1024 * 1024).await.unwrap();
    let json_str = String::from_utf8(bytes.to_vec()).unwrap();

    println!("Response: {}", json_str);
    assert!(json_str.contains("\"object\":\"list\""));
    assert!(json_str.contains("\"model\""));
}

use std::sync::Arc;
