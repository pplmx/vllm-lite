use axum::{Json, extract::State};

use super::types::{BatchResponse, BatchResults, BatchStatus, RequestCounts, SimpleBatchRequest};
use crate::ApiState;
use crate::openai::types::ErrorResponse;

/// Create batch.
/// # Errors
///
/// # Panics
///
/// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
/// Returns `Err` if the operation fails.
pub async fn create_batch(
    State(state): State<ApiState>,
    Json(req): Json<SimpleBatchRequest>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.prompts.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "prompts is required",
                "invalid_request_error",
            )),
        ));
    }

    let id = state
        .batch_manager
        .create_job(
            req.endpoint,
            req.prompts,
            req.model,
            req.max_tokens,
            req.temperature,
        )
        .await;

    let job = state.batch_manager.get_job(&id).await.ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                "batch job missing immediately after creation",
                "internal_error",
            )),
        )
    })?;
    // invariant: SystemTime::now() is always >= UNIX_EPOCH on any platform with a working clock;
    // duration_since cannot underflow.
    let now = i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    )
    .unwrap_or(i64::MAX);

    Ok(Json(BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status: "pending".to_string(),
        created_at: job.created_at,
        expires_at: now + 86400,
        completed_at: None,
        request_counts: Some(RequestCounts {
            total: i32::try_from(job.prompts.len()).unwrap_or(i32::MAX),
            completed: 0,
            failed: 0,
        }),
    }))
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
    // invariant: SystemTime::now() is always >= UNIX_EPOCH on any platform with a working clock;
    // duration_since cannot underflow.
    let now = i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    )
    .unwrap_or(i64::MAX);

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
    async fn test_create_batch_empty_prompts() {
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
    async fn test_create_batch_valid_request() {
        let state = create_test_state();
        let req = SimpleBatchRequest {
            prompts: vec!["Hello".to_string(), "World".to_string()],
            endpoint: BatchEndpoint::Completion,
            model: Some("test-model".to_string()),
            max_tokens: Some(50),
            temperature: Some(0.5),
        };

        let result = create_batch(State(state), Json(req)).await;
        assert!(result.is_ok());
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
