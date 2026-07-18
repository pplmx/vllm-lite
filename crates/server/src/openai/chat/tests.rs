//! Unit tests for the `OpenAI` `/v1/chat/completions` endpoint.
//!
//! Exercises the request-validation and prompt-rendering paths that
//! do not require a live engine channel. The handler-level integration
//! tests (token streaming, error mapping) live in `tests/handlers.rs`.
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
    let result = clean_completion_text(&tokenizer, "hello<|endoftext|><|im_end|><|im_start|>world");
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
        user: None,
        response_format: None,
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
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
    let prompt = build_prompt_from_messages(ChatTemplate::ChatMl, &messages);
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
    let prompt = build_prompt_from_messages(ChatTemplate::ChatMl, &messages);
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
    let prompt = build_prompt_from_messages(ChatTemplate::ChatMl, &messages);
    assert_eq!(
        prompt,
        "<|endoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
    );
}
