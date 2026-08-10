#![allow(clippy::float_cmp)]

//! Unit tests for the `OpenAI` `/v1/chat/completions` endpoint.
//! Exercises the request-validation and prompt-rendering paths that
//! do not require a live engine channel. The handler-level integration
//! tests (token streaming, error mapping) live in `tests/handlers.rs`.
use super::*;
use axum::http::StatusCode;
use vllm_model::tokenizer::Tokenizer;

/// RIL ISS-044: an unknown-context model (no `max_position_embeddings`, no
/// `--max-model-len`) must NOT accept an unbounded `max_tokens`. Before the
/// fix, `check_context_length` returned `Ok(())` when `max_model_len` was
/// `None`, so `max_tokens = i64::MAX` passed every validator and the engine
/// grew the sequence's token list forever — holding the scheduler thread and
/// OOM-ing the process. The request is now rejected with a 400 and a
/// `context_length_exceeded` code.
#[test]
fn test_check_context_length_rejects_unbounded_max_tokens_without_model_len() {
    let err = check_context_length(10, i64::MAX as usize, None).unwrap_err();
    assert_eq!(err.0, StatusCode::BAD_REQUEST);

    let err = check_context_length(10, 1_000_000, None).unwrap_err();
    assert_eq!(err.0, StatusCode::BAD_REQUEST);

    // A bounded request is fine even with no known context...
    assert!(check_context_length(10, 8192, None).is_ok());
    assert!(check_context_length(10, 100, None).is_ok());

    // ...and an explicit max_model_len restores the normal context gate
    // (prompt + max_tokens must fit), applying whether or not a ceiling would
    // otherwise apply.
    assert!(check_context_length(10, 100, Some(8192)).is_ok());
    let err = check_context_length(10, 9000, Some(8192)).unwrap_err();
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(
        check_context_length(10, 900_000, Some(1_000_000)).is_ok(),
        "an explicit max_model_len must win over the ceiling (900k fits a 1M context)"
    );
}

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

/// Regression (RIL ISS-035): the single-shot streaming path must emit a
/// well-formed JSON chunk (empty content) for skipped tokens instead of a
/// bare `data: ` SSE event. Pin the wire shape the skipped-token branch now
/// produces: valid JSON with `choices[0].delta.content == ""`.
#[test]
fn test_skipped_token_chunk_is_valid_json_with_empty_content() {
    let chunk = ChatChunk::new(
        "chatcmpl-stream".to_string(),
        "test-model".to_string(),
        ChatChunkChoice {
            index: 0,
            delta: ChatMessage {
                role: "assistant".to_string(),
                content: String::new(),
                name: None,
            },
            finish_reason: None,
            logprobs: None,
        },
    );
    let json = serde_json::to_string(&chunk).expect("chunk must serialize");
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("skipped-token SSE data must be valid JSON");
    assert_eq!(
        parsed["choices"][0]["delta"]["content"],
        serde_json::Value::String(String::new()),
        "skipped-token chunk must carry empty content"
    );
    assert_eq!(
        parsed["choices"][0]["delta"]["role"],
        serde_json::Value::String("assistant".to_string())
    );
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
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
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

// =============================================================================
// P39 v0.x wire-type engine wire-through — unit tests for
// `build_chat_choice`. Pin the per-candidate ChatChoice assembly
// contract (index mapping, finish_reason string, role/content split,
// logprobs pass-through) so future refactors (e.g. switching to a
// length-penalty variant, dropping the `name: None` field) trip the
// suite.
// =============================================================================

fn chat_sampled(token: vllm_traits::TokenId, logprob: f32) -> vllm_traits::SampledToken {
    vllm_traits::SampledToken {
        token,
        logprob,
        top_logprobs: Vec::new(),
    }
}

#[test]
fn test_build_chat_choice_index_zero_for_first_candidate() {
    // P39: candidate 0's ChatChoice.index must be 0 (matches
    // OpenAI's 0-based convention for `n > 1` responses).
    let tokenizer = test_tokenizer();
    let tokens = vec![chat_sampled(10, -1.0)];
    let choice = build_chat_choice(
        &tokenizer,
        &tokens,
        vllm_traits::FinishReason::Stop,
        0,
        None,
        None,
    );
    assert_eq!(choice.index, 0);
}

#[test]
fn test_build_chat_choice_index_n_for_later_candidate() {
    // P39: candidate i's ChatChoice.index must be i (matches
    // OpenAI's 0-based convention). Pin with i = 5 to confirm
    // the index is forwarded verbatim (not clamped to 0..2 or
    // similar default-path artifact).
    let tokenizer = test_tokenizer();
    let tokens = vec![chat_sampled(20, -0.5)];
    let choice = build_chat_choice(
        &tokenizer,
        &tokens,
        vllm_traits::FinishReason::Length,
        5,
        None,
        None,
    );
    assert_eq!(choice.index, 5);
}

#[test]
fn test_build_chat_choice_finish_reason_length_maps_to_length_string() {
    // P39: each candidate's `finish_reason` MUST be the OpenAI
    // wire string ("length" or "stop") matching the engine's
    // `FinishReason`. Mirrors the single-shot `handle_chat` mapping
    // — keeps chat + chat-n consistent.
    let tokenizer = test_tokenizer();
    let tokens = vec![chat_sampled(10, -1.0)];
    let choice = build_chat_choice(
        &tokenizer,
        &tokens,
        vllm_traits::FinishReason::Length,
        0,
        None,
        None,
    );
    assert_eq!(choice.finish_reason.as_deref(), Some("length"));
}

#[test]
fn test_build_chat_choice_finish_reason_stop_and_cancelled_map_to_stop_string() {
    // P39: BOTH `FinishReason::Stop` AND `FinishReason::Cancelled`
    // map to the OpenAI wire string `"stop"`. Cancellation is an
    // internal detail — the client only sees `"stop"`. Pin both
    // cases explicitly so a future refactor that distinguishes them
    // in the wire shape trips the suite (and we'd want a spec
    // change before that).
    let tokenizer = test_tokenizer();
    let tokens = vec![chat_sampled(10, -1.0)];

    let choice_stop = build_chat_choice(
        &tokenizer,
        &tokens,
        vllm_traits::FinishReason::Stop,
        0,
        None,
        None,
    );
    assert_eq!(choice_stop.finish_reason.as_deref(), Some("stop"));

    let choice_cancelled = build_chat_choice(
        &tokenizer,
        &tokens,
        vllm_traits::FinishReason::Cancelled,
        0,
        None,
        None,
    );
    assert_eq!(
        choice_cancelled.finish_reason.as_deref(),
        Some("stop"),
        "Cancelled must map to 'stop' on the wire (internal detail)"
    );
}

#[test]
fn test_build_chat_choice_role_is_assistant_and_logprobs_omitted_when_not_asked() {
    // P39: chat-specific response shape — each ChatChoice has
    // `message: { role: "assistant", content }` (NOT `text` like
    // completions). Pin both:
    // 1. `message.role` is always "assistant".
    // 2. `logprobs` is `None` when the request did NOT ask for
    //    logprobs (OpenAI spec: omit the field unless asked).
    let tokenizer = test_tokenizer();
    let tokens = vec![chat_sampled(10, -1.0), chat_sampled(11, -1.0)];
    let choice = build_chat_choice(
        &tokenizer,
        &tokens,
        vllm_traits::FinishReason::Stop,
        2,
        None, // logprobs NOT requested
        None,
    );
    assert_eq!(choice.message.role, "assistant");
    assert!(
        choice.logprobs.is_none(),
        "logprobs must be None when req.logprobs != Some(true); got {:?}",
        choice.logprobs
    );
    // `message.name` is None (OpenAI chat completion messages do
    // not carry a `name` for assistant replies — it's a user-side
    // convention only).
    assert!(choice.message.name.is_none());
}
