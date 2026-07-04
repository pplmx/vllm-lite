use serde::{Deserialize, Serialize};

/// Token usage statistics for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

impl Usage {
    #[must_use]
    pub fn new(prompt: usize, completion: usize) -> Self {
        let prompt = i64::try_from(prompt).unwrap_or(0);
        let completion = i64::try_from(completion).unwrap_or(0);
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// Error details following `OpenAI` API error format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

/// Error response wrapper following `OpenAI` API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

impl ErrorResponse {
    #[must_use]
    pub fn new(message: &str, error_type: &str) -> Self {
        Self {
            error: ErrorDetail {
                message: message.to_string(),
                error_type: error_type.to_string(),
                code: None,
            },
        }
    }
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

/// Request body for chat completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<i64>,
    pub stream: Option<bool>,
    pub n: Option<i64>,
    pub stop: Option<Vec<String>>,
}

/// A choice in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Response from chat completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatResponse {
    #[must_use]
    pub fn new(id: String, model: String, choices: Vec<ChatChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
            model,
            choices,
            usage,
        }
    }
}

/// `ChatChunkChoice`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    pub index: i32,
    pub delta: ChatMessage,
    pub finish_reason: Option<String>,
}

/// `ChatChunk`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

impl ChatChunk {
    #[must_use]
    pub fn new(id: String, model: String, choice: ChatChunkChoice) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
            model,
            choices: vec![choice],
        }
    }
}

/// Request body for text completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i64>,
    pub stream: Option<bool>,
    pub n: Option<i64>,
    pub stop: Option<Vec<String>>,
}

/// `CompletionChoice`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: i32,
    pub finish_reason: Option<String>,
}

/// Response from text completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    #[must_use]
    pub fn new(id: String, model: String, choices: Vec<CompletionChoice>, usage: Usage) -> Self {
        Self {
            id,
            object: "text_completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
            model,
            choices,
            usage,
        }
    }
}

/// Request body for embeddings endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
}

/// Embedding: single embedding item in an embeddings response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: i32,
}

/// Deprecated alias for [`Embedding`].
#[deprecated(since = "0.20.0", note = "use Embedding instead")]
pub type EmbeddingData = Embedding;

/// Response from embeddings endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<Embedding>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingsResponse {
    #[must_use]
    pub fn new(embeddings: Vec<Vec<f32>>, model: String) -> Self {
        let items: Vec<Embedding> = embeddings
            .into_iter()
            .enumerate()
            .map(|(i, e)| Embedding {
                object: "embedding".to_string(),
                embedding: e,
                index: i32::try_from(i).unwrap_or(0),
            })
            .collect();

        let total_tokens: i64 = items
            .iter()
            .map(|d| i64::try_from(d.embedding.len()).unwrap_or(0))
            .sum();

        Self {
            object: "list".to_string(),
            data: items,
            model,
            usage: Usage::new(usize::try_from(total_tokens).unwrap_or(0), 0),
        }
    }
}
