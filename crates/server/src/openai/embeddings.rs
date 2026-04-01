use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use crate::ApiState;
use super::types::*;

pub async fn embeddings(
    State(_state): State<ApiState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("model is required", "invalid_request_error")),
        ));
    }
    if req.input.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("input is required", "invalid_request_error")),
        ));
    }
    
    // TODO: 实现真正的 embedding 生成
    // 暂时返回占位数据
    
    let embedding_dim = 512; // 需要从 model 获取
    let embeddings: Vec<Vec<f32>> = req.input
        .iter()
        .map(|_| vec![0.0; embedding_dim])
        .collect();

    Ok(Json(EmbeddingsResponse::new(embeddings, req.model)).into_response())
}