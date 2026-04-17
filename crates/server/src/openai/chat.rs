#![allow(dead_code)]

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse,
        sse::{Event, Sse},
    },
};
use futures::stream;
use std::convert::Infallible;
use tokio::sync::mpsc;

use super::types::*;
use crate::ApiState;

const SPECIAL_TOKENS_TO_SKIP: &[&str] = &["<|endoftext|>", "<|im_end|>", "<|im_start|>"];

fn should_skip_token_text(text: &str) -> bool {
    text.is_empty() || SPECIAL_TOKENS_TO_SKIP.contains(&text)
}

fn clean_completion_text(text: &str) -> String {
    let mut result = text.to_string();
    for token in SPECIAL_TOKENS_TO_SKIP {
        result = result.replace(*token, "");
    }
    result.trim().to_string()
}

pub fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(im_start);
                prompt.push_str("system\n");
                prompt.push_str(&msg.content);
                prompt.push_str(im_end);
                prompt.push('\n');
            }
            "user" => {
                prompt.push_str(im_start);
                prompt.push_str("user\n");
                prompt.push_str(&msg.content);
                prompt.push_str(im_end);
                prompt.push('\n');
            }
            "assistant" => {
                prompt.push_str(im_start);
                prompt.push_str("assistant\n");
                prompt.push_str(&msg.content);
                prompt.push_str(im_end);
                prompt.push('\n');
            }
            _ => {}
        }
    }

    prompt.push_str(im_start);
    prompt.push_str("assistant\n");
    prompt
}

#[allow(dead_code)]
pub fn validate_chat_request(
    req: &ChatRequest,
) -> Result<(), (axum::http::StatusCode, Json<ErrorResponse>)> {
    if req.model.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "model is required",
                "invalid_request_error",
            )),
        ));
    }
    if req.messages.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "messages is required",
                "invalid_request_error",
            )),
        ));
    }
    Ok(())
}

#[allow(dead_code)]
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

    let (response_tx, mut response_rx) = mpsc::channel(64);

    state
        .engine_tx
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

    let completion_text = clean_completion_text(&state.tokenizer.decode(&tokens));
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

#[allow(dead_code)]
pub(crate) async fn chat_completions(
    State(state): State<ApiState>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, (axum::http::StatusCode, Json<ErrorResponse>)> {
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

        let (response_tx, response_rx) = mpsc::channel(64);

        state
            .engine_tx
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
                        if should_skip_token_text(&text) {
                            return Some((Ok::<Event, Infallible>(Event::default().data("")), rx));
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
                        let data =
                            serde_json::to_string(&chunk).expect("Failed to serialize chat chunk");
                        Some((Ok(Event::default().data(data)), rx))
                    }
                    None => {
                        // Channel closed - could be normal completion or client disconnect
                        // With bounded channel, if send fails due to backpressure, we log it
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
                        let data =
                            serde_json::to_string(&chunk).expect("Failed to serialize chat chunk");
                        Some((Ok(Event::default().data(format!("{data}\n\n[DONE]"))), rx))
                    }
                }
            }
        });

        return Ok(Sse::new(Box::pin(stream)).into_response());
    }

    // 非流式 - 返回普通 JSON
    let response = handle_chat(&state, req).await?;
    Ok(Json(response).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn test_should_skip_token_text_empty() {
        assert!(should_skip_token_text(""));
    }

    #[test]
    fn test_should_skip_token_text_eos() {
        assert!(should_skip_token_text("<|endoftext|>"));
    }

    #[test]
    fn test_should_skip_token_text_im_end() {
        assert!(should_skip_token_text("<|im_end|>"));
    }

    #[test]
    fn test_should_skip_token_text_im_start() {
        assert!(should_skip_token_text("<|im_start|>"));
    }

    #[test]
    fn test_should_skip_token_text_normal() {
        assert!(!should_skip_token_text("hello"));
        assert!(!should_skip_token_text("gypt"));
        assert!(!should_skip_token_text(" world"));
    }

    #[test]
    fn test_clean_completion_text_removes_eos() {
        let result = clean_completion_text("gyptabo<|endoftext|>");
        assert_eq!(result, "gyptabo");
    }

    #[test]
    fn test_clean_completion_text_removes_im_end() {
        let result = clean_completion_text("hi<|im_end|>world");
        assert_eq!(result, "hiworld");
    }

    #[test]
    fn test_clean_completion_text_removes_all_special() {
        let result = clean_completion_text("hello<|endoftext|><|im_end|><|im_start|>world");
        assert_eq!(result, "helloworld");
    }

    #[test]
    fn test_clean_completion_text_trims_whitespace() {
        let result = clean_completion_text("  hello  ");
        assert_eq!(result, "hello");
    }

    fn create_test_request(model: &str, messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: Some(100),
            stream: None,
            n: None,
            stop: None,
        }
    }

    #[test]
    fn test_validate_chat_request_valid() {
        let req = create_test_request(
            "test-model",
            vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
            }],
        );
        assert!(validate_chat_request(&req).is_ok());
    }

    #[test]
    fn test_validate_chat_request_empty_model() {
        let req = create_test_request(
            "",
            vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
            }],
        );
        let result = validate_chat_request(&req);
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_validate_chat_request_empty_messages() {
        let req = create_test_request("test-model", vec![]);
        let result = validate_chat_request(&req);
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_build_prompt_from_messages_user_only() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
        }];
        let prompt = build_prompt_from_messages(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_build_prompt_from_messages_system_and_user() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: None,
            },
        ];
        let prompt = build_prompt_from_messages(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_build_prompt_from_messages_with_assistant() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                name: None,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: "How are you?".to_string(),
                name: None,
            },
        ];
        let prompt = build_prompt_from_messages(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
