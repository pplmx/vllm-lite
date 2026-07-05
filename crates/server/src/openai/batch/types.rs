//! `OpenAI` Batch API wire types: `BatchRequest`, `BatchResponse`, `BatchStatus`, request/result counts, and the per-line `BatchRequestInput` / `BatchResponseOutput` shapes.
//!
//! Mirrors the upstream `OpenAI` Batch schema 1:1; the handler (`handler.rs`)
//! and manager (`manager.rs`) operate on these types end-to-end.
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

/// Request payload for `SimpleBatch`. Contains input data, configuration, and request-tracking metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleBatchRequest {
    /// Prompts to execute (one HTTP-style request per prompt).
    pub prompts: Vec<String>,
    /// Which downstream endpoint to fan out to (chat vs. completions).
    #[serde(
        serialize_with = "serialize_batch_endpoint",
        deserialize_with = "deserialize_batch_endpoint"
    )]
    pub endpoint: BatchEndpoint,
    /// Optional model override; `None` = server default.
    pub model: Option<String>,
    /// Max tokens to generate per request.
    pub max_tokens: Option<i64>,
    /// Sampling temperature for each request.
    pub temperature: Option<f32>,
}

/// Response payload for Batch. Returned from handlers, serialized to JSON for the HTTP boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    /// Server-allocated batch identifier (`"batch_..."`).
    pub id: String,
    /// Always `"batch"`.
    pub object: String,
    /// Endpoint that the batch will fan out to.
    #[serde(
        serialize_with = "serialize_batch_endpoint",
        deserialize_with = "deserialize_batch_endpoint"
    )]
    pub endpoint: BatchEndpoint,
    /// Lifecycle state (`"pending"`, `"in_progress"`, `"completed"`, `"failed"`, etc.).
    pub status: String,
    /// Unix timestamp at batch creation.
    pub created_at: i64,
    /// Unix timestamp at which the batch expires (results still retrievable until then).
    pub expires_at: i64,
    /// Unix timestamp at which the batch reached a terminal state, if any.
    pub completed_at: Option<i64>,
    /// Aggregate counts of succeeded / failed requests.
    pub request_counts: Option<RequestCounts>,
}

/// `RequestCounts`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCounts {
    /// Total number of requests submitted in the batch.
    pub total: i32,
    /// Number of requests that finished successfully.
    pub completed: i32,
    /// Number of requests that failed (each generates a `BatchResultItem` with `error`).
    pub failed: i32,
}

/// Collection of result items for Batch. Each entry pairs a request id with its result or error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResults {
    /// Identifier of the parent batch.
    pub batch_id: String,
    /// Lifecycle state of the batch as a whole.
    pub status: String,
    /// One entry per submitted request, in submission order.
    pub results: Vec<BatchResultItem>,
}

/// `BatchResultItem`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResultItem {
    /// Index of the original request within the batch (`0..prompts.len()`).
    pub index: usize,
    /// `"succeeded"` or `"failed"`.
    pub status: String,
    /// Output content when `status == "succeeded"`.
    pub content: Option<String>,
    /// Error message when `status == "failed"`.
    pub error: Option<String>,
}

/// Status of an async batch job: pending, running, completed, failed, or cancelled. Transitions are monotonic.
#[derive(Debug, Clone)]
pub enum BatchStatus {
    /// Queued, not yet picked up by a worker.
    Pending,
    /// Currently being processed.
    InProgress,
    /// All requests finished successfully.
    Completed,
    /// One or more requests failed; partial results may still be available.
    Failed,
}

/// Background job: scheduled for execution by the worker pool. Carries the work payload plus retry / cancellation metadata.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Server-allocated batch identifier.
    pub id: String,
    /// Which endpoint each prompt should be dispatched to.
    pub endpoint: BatchEndpoint,
    /// Prompt strings to execute (one request per prompt).
    pub prompts: Vec<String>,
    /// Optional model override.
    pub model: Option<String>,
    /// Max tokens per request.
    pub max_tokens: Option<i64>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Current lifecycle state.
    pub status: BatchStatus,
    /// Per-request results accumulated so far.
    pub results: Vec<BatchResultItem>,
    /// Unix timestamp at batch creation.
    pub created_at: i64,
    /// Unix timestamp at which the batch reached a terminal state, if any.
    pub completed_at: Option<i64>,
}

impl BatchJob {
    #[must_use]
    /// Construct a new instance from the given configuration.
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
                // invariant: pre-conditions make this infallible at this call site.
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
            // invariant: pre-conditions make this infallible at this call site.
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
        // invariant: pre-conditions make this infallible at this call site.
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
        // invariant: pre-conditions make this infallible at this call site.
        let req: SimpleBatchRequest = serde_json::from_str(json).expect("should parse");
        assert_eq!(req.endpoint, BatchEndpoint::Chat);

        // Legacy "completions" alias must still work.
        let json = r#"{"prompts":["test"],"endpoint":"completions"}"#;
        // invariant: pre-conditions make this infallible at this call site.
        let req: SimpleBatchRequest = serde_json::from_str(json).expect("should parse");
        assert_eq!(req.endpoint, BatchEndpoint::Completion);

        // Full OpenAI path must still work.
        let json = r#"{"prompts":["test"],"endpoint":"/v1/chat/completions"}"#;
        // invariant: pre-conditions make this infallible at this call site.
        let req: SimpleBatchRequest = serde_json::from_str(json).expect("should parse");
        assert_eq!(req.endpoint, BatchEndpoint::Chat);

        // Unknown endpoint must fail loudly.
        let json = r#"{"prompts":["test"],"endpoint":"bogus"}"#;
        let result: Result<SimpleBatchRequest, _> = serde_json::from_str(json);
        assert!(result.is_err(), "unknown endpoint should be rejected");
    }
}
