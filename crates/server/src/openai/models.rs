//! `OpenAI` Models endpoint: `GET /v1/models` (list) and
//! `GET /v1/models/{id}` (retrieve one). Returns the loaded model id(s) so
//! clients can confirm what's deployed.
#![allow(clippy::module_name_repetitions)]
use crate::ApiState;
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use super::types::ErrorResponse;

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
    /// Maximum context length in tokens, exposed so OpenAI-style
    /// clients can size their prompts before sending. `None` when
    /// the loaded model did not declare `max_position_embeddings`
    /// (stub models, some GGUF variants). Production-readiness §4:
    /// the field is a *capacity hint*, not a guarantee — clients
    /// must still handle the `400 context_length_exceeded` error
    /// on race / config-drift conditions.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_model_len: Option<usize>,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

/// OpenAI-compatible `GET /v1/models` handler.
///
/// Returns a single-element list describing the currently loaded
/// model. If the tokenizer does not expose a model name, the
/// placeholder `"unknown"` is reported so clients always receive a
/// well-formed response.
///
/// `async` is currently unused (`axum::Json::into_response` is sync);
/// the keyword is kept so the handler signature matches the rest of
/// the routes and to leave room for future async work (e.g. dynamic
/// model listing once multi-model serving is wired).
#[allow(clippy::unused_async)]
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
            // RIL ISS-098: report when this server instance serves the
            // model entry (vLLM's convention), not a stale hardcoded
            // epoch (1_700_000_000 = 2023-11-14) that implied a fixed
            // creation date for every model on every deployment.
            created: crate::util::time::unix_now_secs(),
            owned_by: "vllm-lite".to_string(),
            max_model_len: state.max_model_len,
        }],
    };

    Json(response).into_response()
}

/// OpenAI-compatible `GET /v1/models/{id}` handler (RIL ISS-106).
///
/// Resolves a single model by id — the id the list endpoint just
/// advertised. `OpenAI` clients / SDKs commonly follow up a model list
/// with `GET /v1/models/<id>` to confirm availability and pull per-model
/// metadata; before this route existed such requests hit axum's default
/// empty `404` (no `OpenAI` error body). Returns the same single-element
/// payload shape as the list endpoint when the id matches the served
/// model, and a `404 model_not_found` `OpenAI` error otherwise.
#[allow(clippy::unused_async)]
pub async fn model_by_id_handler(
    Path(model_id): Path<String>,
    State(state): State<ApiState>,
) -> Response {
    let model_name = state
        .tokenizer
        .model_name()
        .unwrap_or_else(|| "unknown".to_string());

    if model_id != model_name {
        let body = ErrorResponse::with_code(
            &format!("The model '{model_id}' does not exist or you do not have access to it"),
            "invalid_request_error",
            "model_not_found",
        );
        return (StatusCode::NOT_FOUND, Json(body)).into_response();
    }

    let response = ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: model_name,
            object: "model".to_string(),
            created: crate::util::time::unix_now_secs(),
            owned_by: "vllm-lite".to_string(),
            max_model_len: state.max_model_len,
        }],
    };
    Json(response).into_response()
}
