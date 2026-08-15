//! `OpenAI` Embeddings endpoint: `POST /v1/embeddings`. Tokenise the input list and return one embedding vector per input.
use super::types::{EmbeddingsRequest, EmbeddingsResponse, ErrorResponse};
use crate::ApiState;
use axum::{Json, extract::State, response::IntoResponse};
use tokio::sync::mpsc;
use vllm_core::types::EngineMessage;

/// Hard cap on the number of input elements a single `POST /v1/embeddings`
/// request may carry (RIL ISS-068 / TASK-081).
///
/// Each element yields a full model-dimension vector that is materialised,
/// kept in memory, and serialized; with no cap (the generation siblings
/// bound fan-out via `n <= 8` / `best_of <= 20`) a single 1 MiB body of
/// short strings would amplify into an unbounded number of dense vectors.
/// Matches the `OpenAI` embeddings API's documented batch bound (an array
/// of up to 2,048 strings per request).
const MAX_EMBEDDINGS_INPUTS: usize = 2048;

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

    // RIL ISS-068 / TASK-081: per-element + count + context validation,
    // mirroring the sibling boundary checks on chat/completions.
    //
    // Empty / whitespace-only elements reject the whole request — the sync
    // completions contract (`req.prompt.is_empty()` -> 400) applied per
    // element. A single blank string would otherwise encode to a zero-token
    // embed that every sibling path explicitly forbids.
    if req.input.iter().any(|text| text.trim().is_empty()) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "input elements must not be empty or whitespace-only",
                "invalid_request_error",
            )),
        ));
    }

    // Bound the element count — each element is a full model-dimension
    // vector; without a cap a 1 MiB body of tiny strings amplifies into an
    // unbounded response (the `n <= 8` fan-out bound on the siblings).
    if req.input.len() > MAX_EMBEDDINGS_INPUTS {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "too many input elements (maximum is 2,048)",
                "invalid_request_error",
            )),
        ));
    }

    let input_tokens: Vec<Vec<u32>> = req
        .input
        .iter()
        .map(|text| state.tokenizer.encode(text))
        .collect();

    // RIL ISS-068 / TASK-081 (ISS-044 class): reject an input whose token
    // count exceeds the model's context length instead of forcing a
    // full-length embed forward. `check_context_length(prompt, 0, ...)`
    // with `max_tokens = 0` reduces the gate to exactly
    // `prompt_tokens <= max_model_len` (the `max_model_len = None` branch
    // is a no-op for `max_tokens = 0`, which is fine — a stub model with no
    // known context can't be gated, matching the generation paths).
    for (i, tokens) in input_tokens.iter().enumerate() {
        if let Err((_, json)) =
            super::chat::check_context_length(tokens.len(), 0, state.max_model_len)
        {
            return Err((
                axum::http::StatusCode::BAD_REQUEST,
                Json(ErrorResponse::with_code(
                    &format!("input[{i}] {}", json.error.message),
                    "invalid_request_error",
                    "context_length_exceeded",
                )),
            ));
        }
    }

    let (response_tx, mut rx) = mpsc::unbounded_channel::<Vec<Vec<f32>>>();

    state
        .engine_tx
        .try_send(EngineMessage::GetEmbeddings {
            input_tokens,
            response_tx,
        })
        .map_err(|e| super::chat::map_engine_send_error(&e))?;

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
