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
                expires_at: crate::util::time::unix_now_secs() + 86_400,
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
        expires_at: crate::util::time::unix_now_secs() + 86_400,
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
}
