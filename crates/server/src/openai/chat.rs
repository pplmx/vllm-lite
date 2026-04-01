use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use std::convert::Infallible;
use std::pin::Pin;
use tokio::sync::mpsc;

use crate::ApiState;
use super::types::*;

fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("System: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            "user" => {
                prompt.push_str("User: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            "assistant" => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            _ => {}
        }
    }
    
    prompt.push_str("Assistant: ");
    prompt
}

fn validate_chat_request(req: &ChatRequest) -> Result<(), (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("model is required", "invalid_request_error")),
        ));
    }
    if req.messages.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("messages is required", "invalid_request_error")),
        ));
    }
    Ok(())
}

async fn handle_chat(
    state: &ApiState,
    req: ChatRequest,
) -> Result<ChatResponse, (axum::http::StatusCode, Json<ErrorResponse>)> {
    validate_chat_request(&req)?;
    
    let prompt = build_prompt_from_messages(&req.messages);

    let prompt_tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens_len = prompt_tokens.len();
    let max_tokens = req.max_tokens.unwrap_or(100) as usize;
    let total_max = prompt_tokens_len + max_tokens;

    let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

    if let Some(temp) = req.temperature {
        request.sampling_params.temperature = temp;
    }

    let (response_tx, mut response_rx) = mpsc::unbounded_channel();

    state.engine_tx
        .send(vllm_core::types::EngineMessage::AddRequest {
            request,
            response_tx,
        })
        .map_err(|_| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new("Engine unavailable", "internal_error")),
            )
        })?;

    let mut tokens = Vec::new();
    while let Some(token) = response_rx.recv().await {
        tokens.push(token);
    }

    let completion_text = state.tokenizer.decode(&tokens);
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: completion_text,
            name: None,
        },
        finish_reason: Some("stop".to_string()),
    };

    let usage = Usage::new(prompt_tokens_len, tokens.len());

    Ok(ChatResponse::new(
        format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        req.model,
        vec![choice],
        usage,
    ))
}

pub async fn chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>>, (axum::http::StatusCode, Json<ErrorResponse>)> {
    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        let prompt = build_prompt_from_messages(&req.messages);
        let prompt_tokens = state.tokenizer.encode(&prompt);
        let max_tokens = req.max_tokens.unwrap_or(100) as usize;
        let total_max = prompt_tokens.len() + max_tokens;
        
        let model = req.model.clone();

        let mut request = vllm_core::types::Request::new(0, prompt_tokens, total_max);

        if let Some(temp) = req.temperature {
            request.sampling_params.temperature = temp;
        }

        let (response_tx, response_rx) = mpsc::unbounded_channel();

        state.engine_tx
            .send(vllm_core::types::EngineMessage::AddRequest {
                request,
                response_tx,
            })
            .map_err(|_| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new("Engine unavailable", "internal_error")),
                )
            })?;

        let tokenizer = state.tokenizer.clone();
        let stream = stream::unfold(response_rx, move |mut rx| {
            let tokenizer = tokenizer.clone();
            let model = model.clone();
            async move {
                match rx.recv().await {
                    Some(token) => {
                        let text = tokenizer.decode(&[token]);
                        if text.is_empty() {
                            return Some((Ok(Event::default().data("")), rx));
                        }
                        let chunk = ChatChunk::new(
                            "chatcmpl-stream".to_string(),
                            model.clone(),
                            ChatChunkChoice {
                                index: 0,
                                delta: ChatMessage {
                                    role: "assistant".to_string(),
                                    content: text,
                                    name: None,
                                },
                                finish_reason: None,
                            },
                        );
                        let data = serde_json::to_string(&chunk).unwrap();
                        Some((Ok(Event::default().data(format!("data: {}\n\n", data))), rx))
                    }
                    None => {
                        let chunk = ChatChunk::new(
                            "chatcmpl-stream".to_string(),
                            model.clone(),
                            ChatChunkChoice {
                                index: 0,
                                delta: ChatMessage {
                                    role: "assistant".to_string(),
                                    content: String::new(),
                                    name: None,
                                },
                                finish_reason: Some("stop".to_string()),
                            },
                        );
                        let data = serde_json::to_string(&chunk).unwrap();
                        Some((Ok(Event::default().data(format!("data: {}\n\ndata: [DONE]\n\n", data))), rx))
                    }
                }
            }
        });
        
        return Ok(Sse::new(Box::pin(stream)));
    }

    // 非流式
    let response = handle_chat(&state, req).await?;
    let data = serde_json::to_string(&response).unwrap();
    let stream = stream::iter(vec![Ok(Event::default().data(data)) as Result<Event, Infallible>]);
    Ok(Sse::new(Box::pin(stream)))
}