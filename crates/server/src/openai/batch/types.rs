use serde::{Deserialize, Serialize};

/// `OpenAI` Batch API endpoint kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchEndpoint {
    /// `/v1/chat/completions` (chat completions).
    Chat,
    /// `/v1/completions` (legacy text completions).
    Completion,
}

impl BatchEndpoint {
    /// Parse from string. Accepts both short names ("chat", "completions")
    /// and full `OpenAI` paths ("/v1/chat/completions", "/v1/completions") for flexibility.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "chat" | "/v1/chat/completions" => Some(Self::Chat),
            "completion" | "completions" | "/v1/completions" => Some(Self::Completion),
            _ => None,
        }
    }

    /// Canonical short string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
        }
    }
}

impl std::fmt::Display for BatchEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

fn deserialize_batch_endpoint<'de, D>(de: D) -> Result<BatchEndpoint, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(de)?;
    BatchEndpoint::parse(&s)
        .ok_or_else(|| serde::de::Error::custom(format!("unknown batch endpoint: {s}")))
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn serialize_batch_endpoint<S>(value: &BatchEndpoint, ser: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    ser.serialize_str(value.as_str())
}

/// `SimpleBatchRequest`: simple batch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleBatchRequest {
    pub prompts: Vec<String>,
    #[serde(
        serialize_with = "serialize_batch_endpoint",
        deserialize_with = "deserialize_batch_endpoint"
    )]
    pub endpoint: BatchEndpoint,
    pub model: Option<String>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f32>,
}

/// `BatchResponse`: batch response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub id: String,
    pub object: String,
    #[serde(
        serialize_with = "serialize_batch_endpoint",
        deserialize_with = "deserialize_batch_endpoint"
    )]
    pub endpoint: BatchEndpoint,
    pub status: String,
    pub created_at: i64,
    pub expires_at: i64,
    pub completed_at: Option<i64>,
    pub request_counts: Option<RequestCounts>,
}

/// `RequestCounts`: request counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCounts {
    pub total: i32,
    pub completed: i32,
    pub failed: i32,
}

/// `BatchResults`: batch results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResults {
    pub batch_id: String,
    pub status: String,
    pub results: Vec<BatchResultItem>,
}

/// `BatchResultItem`: batch result item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResultItem {
    pub index: usize,
    pub status: String,
    pub content: Option<String>,
    pub error: Option<String>,
}

/// `BatchStatus`: batch status.
#[derive(Debug, Clone)]
pub enum BatchStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// `BatchJob`: batch job.
#[derive(Debug, Clone)]
pub struct BatchJob {
    pub id: String,
    pub endpoint: BatchEndpoint,
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
    #[must_use]
    /// Runs the operation.
    /// # Panics
    ///
    /// Panics if a required invariant is violated (e.g. a `None` value is force-unwrapped or an out-of-bounds index is used).
    pub fn new(
        id: String,
        endpoint: BatchEndpoint,
        prompts: Vec<String>,
        model: Option<String>,
        max_tokens: Option<i64>,
        temperature: Option<f32>,
    ) -> Self {
        // invariant: SystemTime::now() is always >= UNIX_EPOCH on any platform with a working clock;
        // duration_since cannot underflow.
        let now = i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Failed to get system time")
                .as_secs(),
        )
        .unwrap_or(i64::MAX);
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
            endpoint: BatchEndpoint::Chat,
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
        assert_eq!(req.endpoint, BatchEndpoint::Completion);
    }

    #[test]
    fn test_batch_response_serialization() {
        let resp = BatchResponse {
            id: "batch_123".to_string(),
            object: "batch".to_string(),
            endpoint: BatchEndpoint::Chat,
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
        assert!(json.contains("\"endpoint\":\"chat\""));
    }

    #[test]
    fn test_batch_job_creation() {
        let job = BatchJob::new(
            "batch_test".to_string(),
            BatchEndpoint::Chat,
            vec!["prompt1".to_string()],
            Some("qwen".to_string()),
            Some(100),
            Some(0.5),
        );
        assert_eq!(job.id, "batch_test");
        assert_eq!(job.endpoint, BatchEndpoint::Chat);
        assert_eq!(job.prompts.len(), 1);
        assert!(matches!(job.status, BatchStatus::Pending));
    }

    #[test]
    fn test_batch_status_default() {
        let job = BatchJob::new(
            "batch_default".to_string(),
            BatchEndpoint::Completion,
            vec![],
            None,
            None,
            None,
        );
        assert!(matches!(job.status, BatchStatus::Pending));
        assert!(job.completed_at.is_none());
    }

    #[test]
    fn batch_endpoint_parse_all_variants() {
        assert_eq!(BatchEndpoint::parse("chat"), Some(BatchEndpoint::Chat));
        assert_eq!(
            BatchEndpoint::parse("/v1/chat/completions"),
            Some(BatchEndpoint::Chat)
        );
        assert_eq!(
            BatchEndpoint::parse("completion"),
            Some(BatchEndpoint::Completion)
        );
        assert_eq!(
            BatchEndpoint::parse("completions"),
            Some(BatchEndpoint::Completion)
        );
        assert_eq!(
            BatchEndpoint::parse("/v1/completions"),
            Some(BatchEndpoint::Completion)
        );
        assert_eq!(BatchEndpoint::parse("embeddings"), None);
        assert_eq!(BatchEndpoint::parse("unknown"), None);
        assert_eq!(BatchEndpoint::parse(""), None);
    }

    #[test]
    fn batch_endpoint_display() {
        assert_eq!(BatchEndpoint::Chat.to_string(), "chat");
        assert_eq!(BatchEndpoint::Completion.to_string(), "completion");
        assert_eq!(BatchEndpoint::Chat.as_str(), "chat");
        assert_eq!(BatchEndpoint::Completion.as_str(), "completion");
    }

    #[test]
    fn batch_endpoint_serde_json_wire_compat() {
        // Existing JSON wire format must still deserialize.
        let json = r#"{"prompts":["test"],"endpoint":"chat","model":"qwen"}"#;
        let req: SimpleBatchRequest = serde_json::from_str(json).expect("should parse");
        assert_eq!(req.endpoint, BatchEndpoint::Chat);

        // Legacy "completions" alias must still work.
        let json = r#"{"prompts":["test"],"endpoint":"completions"}"#;
        let req: SimpleBatchRequest = serde_json::from_str(json).expect("should parse");
        assert_eq!(req.endpoint, BatchEndpoint::Completion);

        // Full OpenAI path must still work.
        let json = r#"{"prompts":["test"],"endpoint":"/v1/chat/completions"}"#;
        let req: SimpleBatchRequest = serde_json::from_str(json).expect("should parse");
        assert_eq!(req.endpoint, BatchEndpoint::Chat);

        // Unknown endpoint must fail loudly.
        let json = r#"{"prompts":["test"],"endpoint":"bogus"}"#;
        let result: Result<SimpleBatchRequest, _> = serde_json::from_str(json);
        assert!(result.is_err(), "unknown endpoint should be rejected");
    }
}
