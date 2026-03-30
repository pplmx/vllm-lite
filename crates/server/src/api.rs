use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio::sync::mpsc;
use vllm_core::types::{EngineMessage, Request};

use crate::ApiState;

pub type EngineHandle = mpsc::UnboundedSender<EngineMessage>;

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    #[allow(dead_code)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    100
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
    let stream = stream::unfold((response_rx, false), move |(mut rx, sent_done)| {
        let tokenizer = tokenizer.clone();
        async move {
            if sent_done {
                return None;
            }
            match rx.recv().await {
                Some(token) => {
                    let text = tokenizer.decode(&[token]);
                    let chunk = CompletionChunk {
                        id: "cmpl-0".to_string(),
                        choices: vec![Choice { text, index: 0 }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap();
                    Some((Ok(Event::default().data(data)), (rx, false)))
                }
                None => {
                    Some((Ok(Event::default().data("[DONE]")), (rx, true)))
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