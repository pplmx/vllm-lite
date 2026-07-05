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
/// - `model` is empty (`BAD_REQUEST`)
/// - `input` is empty (`BAD_REQUEST`)
/// - the engine channel is closed or fails to respond (`SERVICE_UNAVAILABLE`,
///   code `engine_unavailable`)
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

    state
        .engine_tx
        .send(EngineMessage::GetEmbeddings {
            input_tokens,
            response_tx,
        })
        .map_err(|_| {
            (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::with_code(
                    "Engine unavailable",
                    "server_error",
                    "engine_unavailable",
                )),
            )
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

#[cfg(test)]
mod tests {
    use super::*;

    use axum::http::StatusCode;

    fn create_test_state() -> crate::ApiState {
        crate::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
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
        // The test fixture's `engine_tx` is a closed mpsc channel; the handler
        // surfaces that as a 503 SERVICE_UNAVAILABLE with `code = "engine_unavailable"`
        // so clients know the failure is transient and retryable.
        let (status, body) = result.unwrap_err();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            body.error.code.as_deref(),
            Some("engine_unavailable"),
            "error code must be machine-readable"
        );
    }
}
