//! `OpenAI` Batch API axum handlers: `POST /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel`, `GET /v1/batches/{id}/results`.
//!
//! Each handler is a thin wrapper over `BatchManager` plus RBAC + rate-limit middleware.
use axum::{Json, extract::State};

use super::manager::BatchManager;
use super::types::{
    BatchEndpoint, BatchJob, BatchResponse, BatchResults, BatchStatus, RequestCounts,
    SimpleBatchRequest,
};
use crate::ApiState;
use crate::openai::sampling_validation::validate_temperature;
use crate::openai::types::ErrorResponse;

/// Create batch.
///
/// API-01 (technical due diligence): this endpoint used to return
/// `501 Not Implemented` because the project had no background worker
/// to advance a `BatchJob` from `Pending` -> `InProgress` ->
/// `Completed`. That worker now lives in [`super::worker`]: this
/// handler persists the job, spawns the worker, and returns the
/// `200 OK` batch object immediately.
///
/// # Errors
///
/// Returns `400 Bad Request` (`invalid_request_error`) when the
/// request has no prompts to run.
///
/// Must be `async` for axum 0.8 `Handler` trait compatibility. The
/// worker runs in the background; the handler never awaits it.
#[allow(clippy::unused_async)]
pub async fn create_batch(
    State(state): State<ApiState>,
    Json(req): Json<SimpleBatchRequest>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.prompts.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "At least one prompt is required to create a batch",
                "invalid_request_error",
            )),
        ));
    }

    // RIL ISS-065 / TASK-078: inherit the same sampling-param boundary
    // validation the chat + completions endpoints enforce. `temperature`
    // is forwarded verbatim into `SamplingParams` by the worker, so NaN /
    // ±inf / out-of-[0,2] values would silently corrupt sampling
    // (ISS-048); `max_tokens` is coerced with `unwrap_or(100)` in the
    // worker, so `0` would still emit one token and a negative value
    // would silently become 100 (ISS-033). Reject before persisting so an
    // invalid batch never spawns a worker over garbage params.
    validate_temperature(req.temperature)?;
    if let Some(max_tokens) = req.max_tokens
        && max_tokens < 1
    {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "max_tokens must be a positive integer (max_tokens >= 1)",
                "invalid_request_error",
            )),
        ));
    }

    let job_id = state
        .batch_manager
        .create_job(
            req.endpoint,
            req.prompts,
            req.model,
            req.max_tokens,
            req.temperature,
        )
        .await;

    // The manager retains the authoritative copy; hand the worker a clone.
    let job = state.batch_manager.get_job(&job_id).await.ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                "batch was created but could not be read back from the manager",
                "server_error",
            )),
        )
    })?;
    // Deliberately fire-and-forget: the worker owns the job's execution and
    // the handler must return immediately with the batch object. The returned
    // `JoinHandle` is intentionally dropped (JoinHandle is a Future, so use
    // `drop` — clippy::let_underscore_future).
    drop(super::worker::spawn_batch_worker(&state, job));

    Ok(Json(
        batch_to_response(&job_id, &state.batch_manager, || BatchResponse {
            id: job_id.clone(),
            object: "batch".to_string(),
            // Fallback endpoint used only in the (impossible) race where
            // the job vanished right after creation; the Chat marker is
            // arbitrary since no real job exists to describe.
            endpoint: BatchEndpoint::Chat,
            status: "pending".to_string(),
            created_at: crate::util::time::unix_now_secs(),
            expires_at: crate::util::time::unix_now_secs() + 86_400,
            completed_at: None,
            request_counts: None,
        })
        .await,
    ))
}

/// Get batch.
/// # Errors
///
/// Returns `Err` if the operation fails.
pub async fn get_batch(
    State(state): State<ApiState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    state.batch_manager.get_job(&id).await.ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "batch not found",
                "invalid_request_error",
            )),
        )
    })?;

    Ok(Json(
        batch_to_response(&id, &state.batch_manager, || BatchResponse {
            id: id.clone(),
            object: "batch".to_string(),
            endpoint: BatchEndpoint::Chat,
            status: "not_found".to_string(),
            created_at: 0,
            expires_at: crate::util::time::unix_now_secs() + 86_400,
            completed_at: None,
            request_counts: None,
        })
        .await,
    ))
}

/// Cancel batch.
///
/// Sets the job's cooperative cancel flag (the worker stops dispatching
/// new prompts and cancels its in-flight engine sequence) and transitions
/// the status to `cancelled`. Returns the updated batch object.
///
/// # Errors
///
/// Returns `404 batch not found` (`invalid_request_error`) when no job
/// with `id` exists; returns `409 batch already <terminal>` when the job
/// already reached a terminal state.
pub async fn cancel_batch(
    State(state): State<ApiState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let cancelled = state.batch_manager.request_cancel(&id).await;
    if !cancelled {
        // Distinguish missing from already-terminal: a `Cancelled`/terminal
        // job is a client error (nothing left to cancel); a missing id is 404.
        let job = state.batch_manager.get_job(&id).await;
        return match job {
            Some(_) => Err((
                axum::http::StatusCode::CONFLICT,
                Json(ErrorResponse::new(
                    "batch is already in a terminal state",
                    "invalid_request_error",
                )),
            )),
            None => Err((
                axum::http::StatusCode::NOT_FOUND,
                Json(ErrorResponse::new(
                    "batch not found",
                    "invalid_request_error",
                )),
            )),
        };
    }

    Ok(Json(
        batch_to_response(&id, &state.batch_manager, || BatchResponse {
            id: id.clone(),
            object: "batch".to_string(),
            endpoint: BatchEndpoint::Chat,
            status: "cancelled".to_string(),
            created_at: 0,
            expires_at: crate::util::time::unix_now_secs() + 86_400,
            completed_at: Some(crate::util::time::unix_now_secs()),
            request_counts: None,
        })
        .await,
    ))
}

/// Get batch results.
/// # Errors
///
/// Returns `Err` if the operation fails.
pub async fn get_batch_results(
    State(state): State<ApiState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResults>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let job = state.batch_manager.get_job(&id).await.ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "batch not found",
                "invalid_request_error",
            )),
        )
    })?;

    let status = status_str(&job).to_string();
    Ok(Json(BatchResults {
        batch_id: job.id,
        status,
        results: job.results,
    }))
}

/// Run the operation (see signature for params and return type).
/// # Panics
///
/// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
pub async fn list_batches(State(state): State<ApiState>) -> Json<Vec<BatchResponse>> {
    let jobs = state.batch_manager.get_all_jobs().await;

    let responses: Vec<BatchResponse> = jobs
        .into_iter()
        .map(|job| {
            let (completed, failed) = counts(&job);
            let status = status_str(&job).to_string();
            BatchResponse {
                id: job.id,
                object: "batch".to_string(),
                endpoint: job.endpoint,
                status,
                created_at: job.created_at,
                // Report the stored (fixed-at-creation) expiry, not a
                // recomputed `now + window` that would extend retention on
                // every read (RIL ISS-066 / TASK-079).
                expires_at: job.expires_at,
                completed_at: job.completed_at,
                request_counts: Some(RequestCounts {
                    total: i32::try_from(job.prompts.len()).unwrap_or(i32::MAX),
                    completed,
                    failed,
                }),
            }
        })
        .collect();

    Json(responses)
}

/// Build a [`BatchResponse`] from the manager's live copy of a job.
/// `fallback` is only touched in the impossible race where the job
/// vanished between the call and the read-back (it documents the
/// intended shape without panicking).
async fn batch_to_response<F>(job_id: &str, manager: &BatchManager, fallback: F) -> BatchResponse
where
    F: FnOnce() -> BatchResponse,
{
    let Some(job) = manager.get_job(job_id).await else {
        return fallback();
    };
    let (completed, failed) = counts(&job);
    let status = status_str(&job).to_string();
    BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status,
        created_at: job.created_at,
        // Stored expiry — fixed at creation, not `now + window` (RIL
        // ISS-066 / TASK-079).
        expires_at: job.expires_at,
        completed_at: job.completed_at,
        request_counts: Some(RequestCounts {
            total: i32::try_from(job.prompts.len()).unwrap_or(i32::MAX),
            completed,
            failed,
        }),
    }
}

/// OpenAI-compatible status string for a job's lifecycle state.
const fn status_str(job: &BatchJob) -> &'static str {
    match job.status {
        BatchStatus::Pending => "pending",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
        BatchStatus::Cancelled => "cancelled",
    }
}

/// Counted results, matching the status strings the worker writes
/// (`"succeeded"` / `"failed"`).
fn counts(job: &BatchJob) -> (i32, i32) {
    let completed = i32::try_from(
        job.results
            .iter()
            .filter(|r| r.status == "succeeded")
            .count(),
    )
    .unwrap_or(0);
    let failed =
        i32::try_from(job.results.iter().filter(|r| r.status == "failed").count()).unwrap_or(0);
    (completed, failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::batch::types::BatchEndpoint;
    use crate::test_fixtures;

    fn create_test_state() -> crate::ApiState {
        test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
    }

    #[tokio::test]
    async fn test_create_batch_empty_prompts_returns_400() {
        let state = create_test_state();
        let req = SimpleBatchRequest {
            prompts: vec![],
            endpoint: BatchEndpoint::Chat,
            model: Some("test-model".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
        };

        let result = create_batch(State(state), Json(req)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_batch_persists_job_and_returns_pending() {
        let state = create_test_state();
        let manager = std::sync::Arc::clone(&state.batch_manager);
        let req = SimpleBatchRequest {
            prompts: vec!["Hello".to_string(), "World".to_string()],
            endpoint: BatchEndpoint::Completion,
            model: Some("test-model".to_string()),
            max_tokens: Some(50),
            temperature: Some(0.5),
        };

        let result = create_batch(State(state), Json(req)).await;
        let response = result.expect("create_batch must succeed once the worker exists");
        assert_eq!(response.status, "pending");
        assert!(response.id.starts_with("batch_"));
        // The job should be visible to the manager immediately.
        let job = manager.get_job(&response.id).await;
        assert!(job.is_some());
    }

    /// Build a `SimpleBatchRequest` with the given `temperature` /
    /// `max_tokens` and the rest at valid defaults — mirrors the
    /// `chat_request_with_n` / `completion_request_with_n` helpers in
    /// `sampling_validation.rs` so the batch tests read the same way.
    fn batch_request(temperature: Option<f32>, max_tokens: Option<i64>) -> SimpleBatchRequest {
        SimpleBatchRequest {
            prompts: vec!["Hello".to_string()],
            endpoint: BatchEndpoint::Completion,
            model: Some("test-model".to_string()),
            max_tokens,
            temperature,
        }
    }

    // RIL ISS-065 / TASK-078: the batch endpoint must inherit the same
    // boundary validation the chat + completions endpoints enforce —
    // `validate_temperature` (ISS-048: NaN / ±inf / out-of-[0,2] would
    // silently corrupt sampling) and `max_tokens >= 1` (ISS-033: 0 still
    // emits one token, negative silently coerces to 100). Round 59
    // hardened the batch endpoint for rate-limit cost + the ISS-044
    // context-length ceiling; the float/int sampling-param boundary
    // checks were the remaining hardening-parity gap. Rejection must
    // happen BEFORE `create_job` so an invalid request never leaves a
    // phantom pending job.

    #[tokio::test]
    async fn test_create_batch_none_temperature_and_max_tokens_pass() {
        // Omitted fields are the default path — must be accepted
        // (worker falls back to the engine defaults).
        let state = create_test_state();
        let req = batch_request(None, None);
        let result = create_batch(State(state), Json(req)).await;
        // `Json` is `#[must_use]` — an explicit `let _ =` signals we only
        // care that creation succeeded.
        let _ = result.expect("None temperature / max_tokens must pass (engine defaults)");
    }

    #[tokio::test]
    async fn test_create_batch_boundary_temperature_and_min_max_tokens_pass() {
        // temperature = 0.0 / 2.0 (inclusive OpenAI bounds) and
        // max_tokens = 1 (the minimum) must pass like the sync endpoints.
        for temperature in [Some(0.0_f32), Some(2.0_f32)] {
            let state = create_test_state();
            let req = batch_request(temperature, Some(1));
            let result = create_batch(State(state), Json(req)).await;
            let _ = result.unwrap_or_else(|(_, j)| {
                panic!(
                    "boundary temperature {temperature:?} + max_tokens=1 must pass: {}",
                    j.0.error.message
                )
            });
        }
    }

    #[tokio::test]
    async fn test_create_batch_rejects_nan_temperature() {
        let state = create_test_state();
        let req = batch_request(Some(f32::NAN), None);
        let result = create_batch(State(state), Json(req)).await;
        let (status, body) = result.expect_err("NaN temperature must be rejected");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body.0.error.error_type, "invalid_request_error");
        assert!(body.0.error.message.contains("temperature"));
    }

    #[tokio::test]
    async fn test_create_batch_rejects_infinite_temperature() {
        for t in [f32::INFINITY, f32::NEG_INFINITY] {
            let state = create_test_state();
            let req = batch_request(Some(t), None);
            let result = create_batch(State(state), Json(req)).await;
            let (status, _) = result.expect_err("±inf temperature must be rejected");
            assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn test_create_batch_rejects_negative_temperature() {
        let state = create_test_state();
        let req = batch_request(Some(-1.0), None);
        let result = create_batch(State(state), Json(req)).await;
        let (status, _) = result.expect_err("negative temperature must be rejected");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_create_batch_rejects_out_of_range_temperature() {
        let state = create_test_state();
        let req = batch_request(Some(3.0), None);
        let result = create_batch(State(state), Json(req)).await;
        let (status, body) = result.expect_err("temperature > 2 must be rejected");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body.0.error.message.contains("temperature"));
    }

    #[tokio::test]
    async fn test_create_batch_rejects_zero_max_tokens() {
        let state = create_test_state();
        let manager = std::sync::Arc::clone(&state.batch_manager);
        let req = batch_request(None, Some(0));
        let result = create_batch(State(state), Json(req)).await;
        let (status, body) = result.expect_err("max_tokens = 0 must be rejected");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(body.0.error.message.contains("max_tokens"));
        // Rejection must happen before persist — no phantom pending job.
        assert!(
            manager.get_all_jobs().await.is_empty(),
            "rejected batch must not leave a pending job behind"
        );
    }

    #[tokio::test]
    async fn test_create_batch_rejects_negative_max_tokens() {
        let state = create_test_state();
        let manager = std::sync::Arc::clone(&state.batch_manager);
        let req = batch_request(None, Some(-5));
        let result = create_batch(State(state), Json(req)).await;
        let (status, _) = result.expect_err("negative max_tokens must be rejected");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            manager.get_all_jobs().await.is_empty(),
            "rejected batch must not leave a pending job behind"
        );
    }

    #[tokio::test]
    async fn test_create_batch_response_reports_fixed_stored_expires_at() {
        // RIL ISS-066 / TASK-079: the response's `expires_at` must come
        // from the stored job (fixed at creation), not be synthesised as
        // `now + 86_400` — the old behaviour reported a moving target that
        // extended the retention window on every read.
        let state = create_test_state();
        let req = batch_request(None, Some(10));
        let result = create_batch(State(state), Json(req)).await;
        let response = result.expect("valid batch must be created");
        assert_eq!(
            response.expires_at - response.created_at,
            crate::openai::batch::types::DEFAULT_BATCH_RETENTION_SECS,
            "expires_at must be a fixed retention window from creation, not now+window per read"
        );
    }

    #[tokio::test]
    async fn test_get_batch_not_found() {
        let state = create_test_state();
        let result = get_batch(State(state), axum::extract::Path("nonexistent".to_string())).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_list_batches_empty() {
        let state = create_test_state();
        let result = list_batches(State(state)).await;
        assert!(result.0.is_empty());
    }

    #[tokio::test]
    async fn test_cancel_batch_returns_cancelled_response() {
        let state = create_test_state();
        let id = state
            .batch_manager
            .create_job(
                BatchEndpoint::Completion,
                vec!["a".to_string(), "b".to_string()],
                None,
                Some(10),
                None,
            )
            .await;

        let result = cancel_batch(State(state), axum::extract::Path(id.clone())).await;
        let response = result.expect("cancel must succeed");
        assert_eq!(response.status, "cancelled");
    }

    #[tokio::test]
    async fn test_cancel_batch_not_found() {
        let state = create_test_state();
        let result = cancel_batch(State(state), axum::extract::Path("nope".to_string())).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_cancel_batch_already_terminal_conflicts() {
        let state = create_test_state();
        let id = state
            .batch_manager
            .create_job(BatchEndpoint::Chat, vec!["x".to_string()], None, None, None)
            .await;
        state.batch_manager.set_completed(&id).await;

        let result = cancel_batch(State(state), axum::extract::Path(id)).await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, axum::http::StatusCode::CONFLICT);
    }
}
