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
    State(engine_tx): State<EngineHandle>,
    Json(req): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let prompt_tokens: Vec<u32> = req
        .prompt
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| (i + 1) as u32)
        .collect();

    let max_tokens = req.max_tokens;
    let request = Request::new(0, prompt_tokens, max_tokens);

    let (response_tx, response_rx) = mpsc::unbounded_channel();

    engine_tx
        .send(EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .unwrap();

    let stream = stream::unfold((response_rx, false), |(mut rx, sent_done)| async move {
        if sent_done {
            return None;
        }
        match rx.recv().await {
            Some(token) => {
                let chunk = CompletionChunk {
                    id: "cmpl-0".to_string(),
                    choices: vec![Choice {
                        text: format!("token_{}", token),
                        index: 0,
                    }],
                };
                let data = serde_json::to_string(&chunk).unwrap();
                Some((Ok(Event::default().data(data)), (rx, false)))
            }
            None => {
                Some((Ok(Event::default().data("[DONE]")), (rx, true)))
            }
        }
    });

    Sse::new(stream)
}

pub async fn shutdown(State(engine_tx): State<EngineHandle>) -> &'static str {
    let _ = engine_tx.send(EngineMessage::Shutdown);
    "Shutting down"
}