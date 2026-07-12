//! `OpenAI` Embeddings endpoint: `POST /v1/embeddings`. Tokenise the input list and return one embedding vector per input.
use super::types::{EmbeddingsRequest, EmbeddingsResponse, ErrorResponse};
use crate::ApiState;
use axum::{Json, extract::State, response::IntoResponse};
use tokio::sync::mpsc;
use vllm_core::types::EngineMessage;

/// OpenAI-compatible `/v1/embeddings` HTTP handler.
///
/// Encodes each input string, sends an [`EngineMessage::GetEmbeddings`] to the
/// engine, and serializes the returned vectors back into an OpenAI-shaped JSON
/// response.
///
/// # Errors
///
/// Returns `(StatusCode, ErrorResponse)` when:
/// - the loaded model cannot produce meaningful embeddings
///   (`SERVICE_UNAVAILABLE` or `NOT_IMPLEMENTED`, code
///   `embeddings_unsupported`)
/// - `model` is empty (`BAD_REQUEST`)
/// - `input` is empty (`BAD_REQUEST`)
/// - the engine channel is closed or fails to respond (`SERVICE_UNAVAILABLE`,
///   code `engine_unavailable`)
pub async fn embeddings(
    State(state): State<ApiState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
    // Production-readiness §10: refuse with 501 when the loaded
    // model is a stub (or capabilities couldn't be detected).
    // Stub models return all-zero embeddings, which is
    // meaningless noise that clients would mistakenly use as a
    // real signal. The 501 + `embeddings_unsupported` code lets
    // OpenAI-compatible clients distinguish "your model isn't
    // loaded" from "your model is loaded but doesn't support
    // embeddings".
    let Some(caps) = state.arch_capabilities else {
        return Err((
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse::with_code(
                "Embeddings not supported: architecture capabilities could not be detected for the loaded model",
                "server_error",
                "embeddings_unsupported",
            )),
        ));
    };
    if caps.is_stub() {
        return Err((
            axum::http::StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse::with_code(
                "Embeddings not supported: the loaded model is a stub architecture that returns meaningless (all-zero) vectors",
                "server_error",
                "embeddings_unsupported",
            )),
        ));
    }

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

    state
        .engine_tx
        .try_send(EngineMessage::GetEmbeddings {
            input_tokens,
            response_tx,
        })
        .map_err(|e| match e {
            tokio::sync::mpsc::error::TrySendError::Full(_) => (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::with_code(
                    "Engine overloaded; retry with backoff",
                    "server_error",
                    "engine_overloaded",
                )),
            ),
            tokio::sync::mpsc::error::TrySendError::Closed(_) => (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::with_code(
                    "Engine unavailable",
                    "server_error",
                    "engine_unavailable",
                )),
            ),
        })?;

    let embeddings = rx.recv().await.ok_or_else(|| {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse::with_code(
                "Failed to get embeddings from engine",
                "server_error",
                "engine_unavailable",
            )),
        )
    })?;

    Ok(Json(EmbeddingsResponse::new(embeddings, req.model)).into_response())
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// handler file under the 800-line soft cap. They cover the
// validation gates (empty model / empty input → 400) and the
// engine-channel error mapping (closed channel → 503
// `engine_unavailable`).
#[cfg(test)]
mod tests;
