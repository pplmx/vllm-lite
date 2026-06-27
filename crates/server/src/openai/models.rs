use crate::ApiState;
use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde::Serialize;

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

pub async fn models_handler(State(state): State<ApiState>) -> Response {
    let model_name = state
        .tokenizer
        .model_name()
        .unwrap_or_else(|| "unknown".to_string());

    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: model_name,
            object: "model".to_string(),
            created: 1700000000,
            owned_by: "vllm-lite".to_string(),
        }],
    };

    Json(response).into_response()
}
