use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use vllm_core::engine::Engine;
use vllm_core::types::Request;
use vllm_model::fake::FakeModel;

pub type EngineHandle = Arc<Mutex<Engine<FakeModel>>>;

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    100
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub text: String,
    pub tokens: Vec<u32>,
    pub num_tokens: usize,
}

pub async fn completions(
    State(engine): State<EngineHandle>,
    Json(req): Json<CompletionRequest>,
) -> Json<CompletionResponse> {
    let prompt_tokens: Vec<u32> = req
        .prompt
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| (i + 1) as u32)
        .collect();

    let request = Request::new(
        0, // auto-assign ID
        prompt_tokens.clone(),
        prompt_tokens.len() + req.max_tokens,
    );

    let engine_clone = engine.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut eng = engine_clone.lock().unwrap();
        eng.add_request(request);

        let mut generated: Vec<u32> = Vec::new();
        while eng.has_pending() {
            match eng.step() {
                Ok(outputs) => {
                    for (_, token) in outputs {
                        generated.push(token);
                    }
                }
                Err(e) => {
                    eprintln!("Engine error: {}", e);
                    break;
                }
            }
        }
        generated
    })
    .await
    .unwrap();

    let all_tokens: Vec<u32> = prompt_tokens.iter().chain(result.iter()).copied().collect();

    Json(CompletionResponse {
        text: format!("Generated {} tokens", result.len()),
        tokens: all_tokens,
        num_tokens: result.len(),
    })
}
