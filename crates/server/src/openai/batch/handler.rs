//! `OpenAI` Batch API axum handlers: `POST /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel`, `GET /v1/batches/{id}/results`.
//!
//! Each handler is a thin wrapper over `BatchManager` plus RBAC + rate-limit middleware.
use axum::{Json, extract::State};

use super::types::{BatchResponse, BatchResults, BatchStatus, RequestCounts, SimpleBatchRequest};
use crate::ApiState;
use crate::openai::types::ErrorResponse;

/// Create batch.
///
/// API-01 (technical due diligence): the Batch API surface accepts
/// requests, validates them, and persists a `BatchJob` in memory —
/// but the project has no background worker that would advance
/// `BatchJob` state from `Pending` -> `InProgress` -> `Completed`.
/// Without a worker, a successful `create_batch` returns `pending`
/// that never resolves, `get_batch` reports a status that is never
/// updated, and `get_batch_results` returns an empty array forever.
///
/// The honest options are:
///   1. Return `200 OK` and silently leave the job stuck (current
///      behaviour — the technical due diligence calls this out as
///      misleadingly compatible with the `OpenAI` Batch API).
///   2. Return `501 Not Implemented` so SDKs and operators see an
///      explicit, machine-readable "this endpoint exists but the
///      server does not implement it" signal.
///   3. Implement the worker (a real tokio task that drains the
///      job's prompts into the engine and updates state).
///
/// We choose (2) for now: the handler still validates the request
/// shape and returns the missing-piece error code so SDKs can
/// distinguish "your request was malformed" from "the server is
/// not yet capable". Once a worker lands, this handler can flip
/// back to (1) without changing the surrounding types.
///
/// `GET /v1/batches/{id}` and `GET /v1/batches/{id}/results`
/// continue to return whatever state the job has today, since those
/// endpoints are useful for inspecting legacy or imported jobs and
/// are also wired through `BatchManager` (read-only).
///
/// # Errors
///
/// Always returns `501 Not Implemented` (error code `server_error`)
/// because the Batch API executor is not yet implemented. The
/// request is validated for shape but never persisted; callers
/// should retry after the executor ships.
pub async fn create_batch(
    State(_state): State<ApiState>,
    Json(_req): Json<SimpleBatchRequest>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    Err((
        axum::http::StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse::new(
            "Batch API executor is not implemented; the server can \
             persist the job but no worker advances state from \
             pending to completed. Track the implementation in \
             docs/technical-due-diligence/architecture-performance.md#api-01.",
            "server_error",
        )),
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
    let job = state.batch_manager.get_job(&id).await.ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "batch not found",
                "invalid_request_error",
            )),
        )
    })?;

    let status = match job.status {
        BatchStatus::Pending => "pending",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
    };

    let completed =
        i32::try_from(job.results.iter().filter(|r| r.status == "success").count()).unwrap_or(0);
    let failed =
        i32::try_from(job.results.iter().filter(|r| r.status == "error").count()).unwrap_or(0);

    Ok(Json(BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status: status.to_string(),
        created_at: job.created_at,
        expires_at: job.created_at + 86400,
        completed_at: job.completed_at,
        request_counts: Some(RequestCounts {
            total: i32::try_from(job.prompts.len()).unwrap_or(i32::MAX),
            completed,
            failed,
        }),
    }))
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

    let status = match job.status {
        BatchStatus::Pending => "pending",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
    };

    Ok(Json(BatchResults {
        batch_id: job.id,
        status: status.to_string(),
        results: job.results,
    }))
}

/// Run the operation (see signature for params and return type).
/// # Panics
///
/// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
pub async fn list_batches(State(state): State<ApiState>) -> Json<Vec<BatchResponse>> {
    let jobs = state.batch_manager.get_all_jobs().await;
    // See `crate::util::time::unix_now_secs` for the panic-free contract.
    let now = crate::util::time::unix_now_secs();

    let responses: Vec<BatchResponse> = jobs
        .into_iter()
        .map(|job| {
            let status = match job.status {
                BatchStatus::Pending => "pending",
                BatchStatus::InProgress => "in_progress",
                BatchStatus::Completed => "completed",
                BatchStatus::Failed => "failed",
            };
            let completed =
                i32::try_from(job.results.iter().filter(|r| r.status == "success").count())
                    .unwrap_or(0);
            let failed = i32::try_from(job.results.iter().filter(|r| r.status == "error").count())
                .unwrap_or(0);

            BatchResponse {
                id: job.id,
                object: "batch".to_string(),
                endpoint: job.endpoint,
                status: status.to_string(),
                created_at: job.created_at,
                expires_at: now + 86400,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::batch::types::BatchEndpoint;

    fn create_test_state() -> crate::ApiState {
        crate::test_fixtures::api_state(vllm_model::config::Architecture::Qwen3)
    }

    #[tokio::test]
    async fn test_create_batch_empty_prompts_returns_501_until_executor_exists() {
        // API-01: with the executor unimplemented, empty-prompt
        // rejection (which used to be a 400) is now subsumed by the
        // 501 short-circuit. Once the executor lands, restore the
        // empty-prompt -> 400 path by re-introducing the validator
        // *before* the 501 return.
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
        assert_eq!(status, axum::http::StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn test_create_batch_returns_501_until_executor_exists() {
        // API-01: the Batch API surface persists jobs but has no
        // worker to advance them. Until that lands we surface a
        // 501 instead of misleadingly returning a job that will
        // stay pending forever.
        let state = create_test_state();
        let req = SimpleBatchRequest {
            prompts: vec!["Hello".to_string(), "World".to_string()],
            endpoint: BatchEndpoint::Completion,
            model: Some("test-model".to_string()),
            max_tokens: Some(50),
            temperature: Some(0.5),
        };

        let result = create_batch(State(state), Json(req)).await;
        let (status, _) = result.expect_err("create_batch must reject with an error");
        assert_eq!(
            status,
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "create_batch must return 501 until a worker exists"
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
}
