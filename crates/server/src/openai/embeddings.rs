#![allow(dead_code)]

use super::types::*;
use crate::ApiState;
use axum::{Json, extract::State, response::IntoResponse};
use tokio::sync::mpsc;
use vllm_core::types::EngineMessage;

#[allow(dead_code)]
pub(crate) async fn embeddings(
    State(state): State<ApiState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "model is required",
                "invalid_request_error",
            )),
        ));
    }
    if req.input.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "input is required",
                "invalid_request_error",
            )),
        ));
    }

    let input_tokens: Vec<Vec<u32>> = req
        .input
        .iter()
        .map(|text| state.tokenizer.encode(text))
        .collect();

    let (response_tx, mut rx) = mpsc::unbounded_channel::<Vec<Vec<f32>>>();

    let _ = state.engine_tx.send(EngineMessage::GetEmbeddings {
        input_tokens,
        response_tx,
    });

    let embeddings = rx.recv().await.ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                "Failed to get embeddings from engine",
                "internal_error",
            )),
        )
    })?;

    Ok(Json(EmbeddingsResponse::new(embeddings, req.model)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ApiState;
    use axum::http::StatusCode;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use vllm_model::tokenizer::Tokenizer;

    fn create_test_state() -> crate::ApiState {
        use vllm_core::metrics::EnhancedMetricsCollector;
        let (engine_tx, _rx) = mpsc::unbounded_channel();
        crate::ApiState {
            engine_tx,
            tokenizer: Arc::new(Tokenizer::new()),
            batch_manager: Arc::new(crate::openai::batch::manager::BatchManager::new()),
            auth: None,
            health: Arc::new(std::sync::RwLock::new(crate::HealthChecker::new(
                true, true,
            ))),
            metrics: Arc::new(EnhancedMetricsCollector::new()),
        }
    }

    #[tokio::test]
    async fn test_embeddings_empty_model() {
        let state = create_test_state();
        let req = EmbeddingsRequest {
            model: String::new(),
            input: vec!["test input".to_string()],
        };

        let result = embeddings(State(state), Json(req)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_embeddings_empty_input() {
        let state = create_test_state();
        let req = EmbeddingsRequest {
            model: "test-model".to_string(),
            input: vec![],
        };

        let result = embeddings(State(state), Json(req)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_embeddings_multiple_inputs() {
        let state = create_test_state();
        let req = EmbeddingsRequest {
            model: "test-model".to_string(),
            input: vec!["input1".to_string(), "input2".to_string()],
        };

        let result = embeddings(State(state), Json(req)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
