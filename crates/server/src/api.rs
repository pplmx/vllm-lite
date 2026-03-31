use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio::sync::mpsc;
use vllm_core::metrics::MetricsSnapshot;
use vllm_core::types::{EngineMessage, Request};

use crate::ApiState;

pub type EngineHandle = mpsc::UnboundedSender<EngineMessage>;

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    100
}

fn default_stream() -> bool {
    false
}

#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    choices: Vec<Choice>,
}

pub async fn completions(
    State(state): State<ApiState>,
    Json(req): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let prompt_tokens = state.tokenizer.encode(&req.prompt);

    let max_new_tokens = req.max_tokens;
    let total_max_tokens = prompt_tokens.len() + max_new_tokens;
    let request = Request::new(0, prompt_tokens, total_max_tokens);

    let (response_tx, response_rx) = mpsc::unbounded_channel();

    state
        .engine_tx
        .send(EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .expect("Engine channel should be available");

    let tokenizer = state.tokenizer.clone();
    let stream = stream::unfold((response_rx, req.stream, Vec::new()), move |(mut rx, streaming, mut tokens)| {
        let tokenizer = tokenizer.clone();
        async move {
            if !streaming && !tokens.is_empty() {
                let text = tokenizer.decode(&tokens);
                let response = CompletionResponse {
                    id: "cmpl-0".to_string(),
                    choices: vec![Choice { text, index: 0 }],
                };
                let data = serde_json::to_string(&response).unwrap();
                return Some((Ok(Event::default().data(data)), (rx, streaming, tokens)));
            }
            match rx.recv().await {
                Some(token) => {
                    if streaming {
                        let text = tokenizer.decode(&[token]);
                        let chunk = CompletionChunk {
                            id: "cmpl-0".to_string(),
                            choices: vec![Choice { text, index: 0 }],
                        };
                        let data = serde_json::to_string(&chunk).unwrap();
                        Some((Ok(Event::default().data(data)), (rx, streaming, tokens)))
                    } else {
                        tokens.push(token);
                        Some((Ok(Event::default().data("".to_string())), (rx, streaming, tokens)))
                    }
                }
                None => {
                    if !streaming {
                        let text = tokenizer.decode(&tokens);
                        let response = CompletionResponse {
                            id: "cmpl-0".to_string(),
                            choices: vec![Choice { text, index: 0 }],
                        };
                        let data = serde_json::to_string(&response).unwrap();
                        Some((Ok(Event::default().data(data)), (rx, true, tokens)))
                    } else {
                        Some((Ok(Event::default().data("[DONE]")), (rx, true, tokens)))
                    }
                }
            }
        }
    });

    Sse::new(stream)
}

pub async fn shutdown(State(engine_tx): State<EngineHandle>) -> &'static str {
    let _ = engine_tx.send(EngineMessage::Shutdown);
    "Shutting down"
}

pub async fn get_stats(State(state): State<ApiState>) -> Json<MetricsSnapshot> {
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    state
        .engine_tx
        .send(EngineMessage::GetMetrics { response_tx })
        .expect("Engine channel should be available");
    let snapshot = response_rx.recv().await.unwrap_or(MetricsSnapshot {
        tokens_total: 0,
        requests_total: 0,
        avg_latency_ms: 0.0,
        p50_latency_ms: 0.0,
        p90_latency_ms: 0.0,
        p99_latency_ms: 0.0,
        avg_batch_size: 0.0,
        current_batch_size: 0,
    });
    Json(snapshot)
}

pub async fn get_prometheus(State(state): State<ApiState>) -> String {
    let (response_tx, mut response_rx) = mpsc::unbounded_channel();
    state
        .engine_tx
        .send(EngineMessage::GetMetrics { response_tx })
        .expect("Engine channel should be available");
    let m = response_rx.recv().await.unwrap_or(MetricsSnapshot {
        tokens_total: 0,
        requests_total: 0,
        avg_latency_ms: 0.0,
        p50_latency_ms: 0.0,
        p90_latency_ms: 0.0,
        p99_latency_ms: 0.0,
        avg_batch_size: 0.0,
        current_batch_size: 0,
    });
    format!(
        "vllm_tokens_total {}\nvllm_requests_total {}\nvllm_avg_latency_ms {:.2}\nvllm_p50_latency_ms {:.2}\nvllm_p90_latency_ms {:.2}\nvllm_p99_latency_ms {:.2}\nvllm_avg_batch_size {:.2}\nvllm_current_batch_size {}\n",
        m.tokens_total,
        m.requests_total,
        m.avg_latency_ms,
        m.p50_latency_ms,
        m.p90_latency_ms,
        m.p99_latency_ms,
        m.avg_batch_size,
        m.current_batch_size
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_max_tokens() {
        assert_eq!(default_max_tokens(), 100);
    }

    #[test]
    fn test_completion_chunk_serialization() {
        let chunk = CompletionChunk {
            id: "test-id".to_string(),
            choices: vec![Choice {
                text: "hello".to_string(),
                index: 0,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("test-id"));
        assert!(json.contains("hello"));
    }

    #[test]
    fn test_completion_request_defaults() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "test"}"#).unwrap();
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.prompt, "test");
    }

    #[test]
    fn test_completion_request_with_max_tokens() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "test", "max_tokens": 50}"#).unwrap();
        assert_eq!(req.max_tokens, 50);
    }

    #[test]
    fn test_completion_request_invalid_json() {
        let result: Result<CompletionRequest, _> = serde_json::from_str("{invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_request_missing_prompt() {
        let result: Result<CompletionRequest, _> = serde_json::from_str("{}");
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_request_negative_max_tokens() {
        let result: Result<CompletionRequest, _> = serde_json::from_str(r#"{"prompt": "test", "max_tokens": -1}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_completion_request_very_long_max_tokens() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "test", "max_tokens": 100000}"#).unwrap();
        assert_eq!(req.max_tokens, 100000);
    }

    #[test]
    fn test_completion_chunk_empty_choices() {
        let chunk = CompletionChunk {
            id: "test".to_string(),
            choices: vec![],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"choices\":[]"));
    }
}