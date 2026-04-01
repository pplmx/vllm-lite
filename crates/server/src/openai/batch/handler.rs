use axum::{extract::State, Json};

use super::types::*;
use crate::openai::types::ErrorResponse;
use crate::ApiState;

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

    if req.endpoint != "chat" && req.endpoint != "completions" {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "endpoint must be 'chat' or 'completions'",
                "invalid_request_error",
            )),
        ));
    }

    let id = state
        .batch_manager
        .create_job(
            req.endpoint.clone(),
            req.prompts,
            req.model,
            req.max_tokens,
            req.temperature,
        )
        .await;

    let job = state.batch_manager.get_job(&id).await.unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    Ok(Json(BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status: "pending".to_string(),
        created_at: job.created_at,
        expires_at: now + 86400,
        completed_at: None,
        request_counts: Some(RequestCounts {
            total: job.prompts.len() as i32,
            completed: 0,
            failed: 0,
        }),
    }))
}

pub async fn get_batch(
    State(state): State<ApiState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResponse>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let job = state.batch_manager.get_job(&id).await.ok_or((
        axum::http::StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(
            "batch not found",
            "invalid_request_error",
        )),
    ))?;

    let status = match job.status {
        BatchStatus::Pending => "pending",
        BatchStatus::InProgress => "in_progress",
        BatchStatus::Completed => "completed",
        BatchStatus::Failed => "failed",
    };

    let completed = job.results.iter().filter(|r| r.status == "success").count() as i32;
    let failed = job.results.iter().filter(|r| r.status == "error").count() as i32;

    Ok(Json(BatchResponse {
        id: job.id,
        object: "batch".to_string(),
        endpoint: job.endpoint,
        status: status.to_string(),
        created_at: job.created_at,
        expires_at: job.created_at + 86400,
        completed_at: job.completed_at,
        request_counts: Some(RequestCounts {
            total: job.prompts.len() as i32,
            completed,
            failed,
        }),
    }))
}

pub async fn get_batch_results(
    State(state): State<ApiState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<BatchResults>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let job = state.batch_manager.get_job(&id).await.ok_or((
        axum::http::StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(
            "batch not found",
            "invalid_request_error",
        )),
    ))?;

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

pub async fn list_batches(State(state): State<ApiState>) -> Json<Vec<BatchResponse>> {
    let jobs = state.batch_manager.get_all_jobs().await;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let responses: Vec<BatchResponse> = jobs
        .into_iter()
        .map(|job| {
            let status = match job.status {
                BatchStatus::Pending => "pending",
                BatchStatus::InProgress => "in_progress",
                BatchStatus::Completed => "completed",
                BatchStatus::Failed => "failed",
            };
            let completed = job.results.iter().filter(|r| r.status == "success").count() as i32;
            let failed = job.results.iter().filter(|r| r.status == "error").count() as i32;

            BatchResponse {
                id: job.id,
                object: "batch".to_string(),
                endpoint: job.endpoint,
                status: status.to_string(),
                created_at: job.created_at,
                expires_at: now + 86400,
                completed_at: job.completed_at,
                request_counts: Some(RequestCounts {
                    total: job.prompts.len() as i32,
                    completed,
                    failed,
                }),
            }
        })
        .collect();

    Json(responses)
}
