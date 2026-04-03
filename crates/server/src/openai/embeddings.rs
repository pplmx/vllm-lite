use super::types::*;
use crate::ApiState;
use axum::{Json, extract::State, response::IntoResponse};
use tokio::sync::mpsc;
use vllm_core::types::EngineMessage;

pub async fn embeddings(
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

    let embeddings = rx.recv().await
        .ok_or_else(|| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    "Failed to get embeddings from engine",
                    "internal_error"
                ))
            )
        })?;

    Ok(Json(EmbeddingsResponse::new(embeddings, req.model)).into_response())
}
