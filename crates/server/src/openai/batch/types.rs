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
            .unwrap()
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
