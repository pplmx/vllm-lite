//! Shared fixtures for server unit and integration tests.
#![allow(dead_code)]

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vllm_core::metrics::EnhancedMetricsCollector;
use vllm_core::types::{EngineMessage, TokenId};
use vllm_model::config::Architecture;
use vllm_model::tokenizer::Tokenizer;

use crate::ApiState;
use crate::api::EngineHandle;
use crate::health::HealthChecker;
use crate::openai::batch::manager::BatchManager;

/// Build [`ApiState`] with defaults suitable for handler tests.
pub fn api_state(architecture: Architecture) -> ApiState {
    let (engine_tx, _engine_rx) = mpsc::unbounded_channel();
    ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
    }
}

/// Mock engine that replies with the given token stream for each chat request.
pub fn spawn_mock_engine(reply_tokens: Vec<TokenId>) -> (EngineHandle, JoinHandle<()>) {
    let (engine_tx, mut engine_rx) = mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        while let Some(msg) = engine_rx.recv().await {
            if let EngineMessage::AddRequest { response_tx, .. } = msg {
                for token in &reply_tokens {
                    if response_tx.send(*token).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
    (engine_tx, handle)
}

/// [`ApiState`] wired to a mock engine that emits `reply_tokens`.
pub fn api_state_with_mock_engine(
    architecture: Architecture,
    reply_tokens: Vec<TokenId>,
) -> (ApiState, JoinHandle<()>) {
    let (engine_tx, handle) = spawn_mock_engine(reply_tokens);
    let state = ApiState {
        engine_tx,
        tokenizer: Arc::new(Tokenizer::new()),
        architecture,
        batch_manager: Arc::new(BatchManager::new()),
        auth: None,
        health: Arc::new(std::sync::RwLock::new(HealthChecker::new(true, true))),
        metrics: Arc::new(EnhancedMetricsCollector::new()),
    };
    (state, handle)
}
