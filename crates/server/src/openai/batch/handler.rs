use axum::{Json, extract::State};

use super::types::*;
use crate::ApiState;
use crate::openai::types::ErrorResponse;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::batch::manager::BatchManager;
    use std::sync::Arc;
    use vllm_model::tokenizer::Tokenizer;

    fn create_test_state() -> crate::ApiState {
        use vllm_core::metrics::EnhancedMetricsCollector;
        let tokenizer = Tokenizer::new();
        let (engine_tx, _engine_rx) = tokio::sync::mpsc::unbounded_channel();
        crate::ApiState {
            engine_tx,
            tokenizer: Arc::new(tokenizer),
            batch_manager: Arc::new(BatchManager::new()),
            auth: None,
            health: Arc::new(std::sync::RwLock::new(crate::HealthChecker::new(
                true, true,
            ))),
            metrics: Arc::new(EnhancedMetricsCollector::new()),
        }
    }

    #[tokio::test]
    async fn test_create_batch_empty_prompts() {
        let state = create_test_state();
        let req = SimpleBatchRequest {
            prompts: vec![],
            endpoint: "chat".to_string(),
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
    async fn test_create_batch_invalid_endpoint() {
        let state = create_test_state();
        let req = SimpleBatchRequest {
            prompts: vec!["Hello".to_string()],
            endpoint: "invalid".to_string(),
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
            endpoint: "completions".to_string(),
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
