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

fn should_skip_token_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> bool {
    text.is_empty() || tokenizer.is_special_token(text)
}

fn clean_completion_text(tokenizer: &vllm_model::tokenizer::Tokenizer, text: &str) -> String {
    tokenizer.clean_special_tokens(text)
}

pub fn build_prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";
    let bos_token = "<|endoftext|>"; // Qwen3 uses this as BOS

    // Add BOS token at the beginning
    prompt.push_str(bos_token);

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
    let start = std::time::Instant::now();
    let request_id = format!("req_{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

    validate_chat_request(&req)?;

    let prompt = build_prompt_from_messages(&req.messages);
    let prompt_tokens = state.tokenizer.encode(&prompt);
    let prompt_tokens_len = prompt_tokens.len();

    tracing::info!(
        request_id = %request_id,
        prompt_tokens = prompt_tokens_len,
        "Request started"
    );
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

    let raw_decode = state.tokenizer.decode(&tokens);

    let completion_text = clean_completion_text(&state.tokenizer, &raw_decode);

    let duration_ms = start.elapsed().as_millis() as u64;
    let output_tokens_len = tokens.len();

    tracing::info!(
        request_id = %request_id,
        output_tokens = output_tokens_len,
        duration_ms = duration_ms,
        "Request completed"
    );
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
                        if should_skip_token_text(&tokenizer, &text) {
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
    use vllm_model::tokenizer::Tokenizer;

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::new()
    }

    #[test]
    fn test_should_skip_token_text_empty() {
        let tokenizer = test_tokenizer();
        assert!(should_skip_token_text(&tokenizer, ""));
    }

    #[test]
    fn test_should_skip_token_text_eos() {
        let tokenizer = test_tokenizer();
        assert!(should_skip_token_text(&tokenizer, "<|endoftext|>"));
    }

    #[test]
    fn test_should_skip_token_text_im_end() {
        let tokenizer = test_tokenizer();
        assert!(should_skip_token_text(&tokenizer, "<|im_end|>"));
    }

    #[test]
    fn test_should_skip_token_text_im_start() {
        let tokenizer = test_tokenizer();
        assert!(should_skip_token_text(&tokenizer, "<|im_start|>"));
    }

    #[test]
    fn test_should_skip_token_text_normal() {
        let tokenizer = test_tokenizer();
        assert!(!should_skip_token_text(&tokenizer, "hello"));
        assert!(!should_skip_token_text(&tokenizer, "gypt"));
        assert!(!should_skip_token_text(&tokenizer, " world"));
    }

    #[test]
    fn test_clean_completion_text_removes_eos() {
        let tokenizer = test_tokenizer();
        let result = clean_completion_text(&tokenizer, "gyptabo<|endoftext|>");
        assert_eq!(result, "gyptabo");
    }

    #[test]
    fn test_clean_completion_text_removes_im_end() {
        let tokenizer = test_tokenizer();
        let result = clean_completion_text(&tokenizer, "hi<|im_end|>world");
        assert_eq!(result, "hiworld");
    }

    #[test]
    fn test_clean_completion_text_removes_all_special() {
        let tokenizer = test_tokenizer();
        let result =
            clean_completion_text(&tokenizer, "hello<|endoftext|><|im_end|><|im_start|>world");
        assert_eq!(result, "helloworld");
    }

    #[test]
    fn test_clean_completion_text_trims_whitespace() {
        let tokenizer = test_tokenizer();
        let result = clean_completion_text(&tokenizer, "  hello  ");
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
            "<|endoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
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
            "<|endoftext|><|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
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
            "<|endoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
