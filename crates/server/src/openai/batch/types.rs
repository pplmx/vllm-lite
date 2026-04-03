use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleBatchRequest {
    pub prompts: Vec<String>,
    pub endpoint: String, // "chat" 或 "completions"
    pub model: Option<String>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub status: String,
    pub created_at: i64,
    pub expires_at: i64,
    pub completed_at: Option<i64>,
    pub request_counts: Option<RequestCounts>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCounts {
    pub total: i32,
    pub completed: i32,
    pub failed: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResults {
    pub batch_id: String,
    pub status: String,
    pub results: Vec<BatchResultItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResultItem {
    pub index: usize,
    pub status: String,
    pub content: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum BatchStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Clone)]
pub struct BatchJob {
    pub id: String,
    pub endpoint: String,
    pub prompts: Vec<String>,
    pub model: Option<String>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
    pub status: BatchStatus,
    pub results: Vec<BatchResultItem>,
    pub created_at: i64,
    pub completed_at: Option<i64>,
}

impl BatchJob {
    pub fn new(
        id: String,
        endpoint: String,
        prompts: Vec<String>,
        model: Option<String>,
        max_tokens: Option<i64>,
        temperature: Option<f32>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Failed to get system time")
            .as_secs() as i64;
        Self {
            id,
            endpoint,
            prompts,
            model,
            max_tokens,
            temperature,
            status: BatchStatus::Pending,
            results: Vec::new(),
            created_at: now,
            completed_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_batch_request_serialization() {
        let req = SimpleBatchRequest {
            prompts: vec!["Hello".to_string(), "World".to_string()],
            endpoint: "chat".to_string(),
            model: Some("qwen".to_string()),
            max_tokens: Some(100),
            temperature: Some(0.7),
        };
        let json = serde_json::to_string(&req).expect("Failed to serialize batch request");
        assert!(json.contains("\"prompts\""));
        assert!(json.contains("\"endpoint\":\"chat\""));
    }

    #[test]
    fn test_simple_batch_request_deserialization() {
        let json = r#"{"prompts":["test"],"endpoint":"completions"}"#;
        let req: SimpleBatchRequest =
            serde_json::from_str(json).expect("Failed to deserialize batch request");
        assert_eq!(req.prompts.len(), 1);
        assert_eq!(req.endpoint, "completions");
    }

    #[test]
    fn test_batch_response_serialization() {
        let resp = BatchResponse {
            id: "batch_123".to_string(),
            object: "batch".to_string(),
            endpoint: "chat".to_string(),
            status: "pending".to_string(),
            created_at: 1000,
            expires_at: 2000,
            completed_at: None,
            request_counts: Some(RequestCounts {
                total: 10,
                completed: 5,
                failed: 1,
            }),
        };
        let json = serde_json::to_string(&resp).expect("Failed to serialize batch response");
        assert!(json.contains("\"id\":\"batch_123\""));
        assert!(json.contains("\"status\":\"pending\""));
    }

    #[test]
    fn test_batch_job_creation() {
        let job = BatchJob::new(
            "batch_test".to_string(),
            "chat".to_string(),
            vec!["prompt1".to_string()],
            Some("qwen".to_string()),
            Some(100),
            Some(0.5),
        );
        assert_eq!(job.id, "batch_test");
        assert_eq!(job.endpoint, "chat");
        assert_eq!(job.prompts.len(), 1);
        assert!(matches!(job.status, BatchStatus::Pending));
    }

    #[test]
    fn test_batch_status_default() {
        let job = BatchJob::new(
            "batch_default".to_string(),
            "completions".to_string(),
            vec![],
            None,
            None,
            None,
        );
        assert!(matches!(job.status, BatchStatus::Pending));
        assert!(job.completed_at.is_none());
    }
}
