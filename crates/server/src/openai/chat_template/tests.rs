//! Unit tests for the chat-template renderer.
//!
//! Exercises the per-architecture prompt builders used by the
//! `chat/completions` handler:
//!
//! - **ChatML** (Qwen3): wraps `<|im_start|>`/`<|im_end|>` markers
//!   around role-tagged messages.
//! - **Llama3**: uses `<|begin_of_text|>` and `assistant<|end_header_id|>`
//!   sentinels (loose `contains` checks because exact whitespace
//!   formatting is allowed to drift).
//! - **MistralInst**: `[INST]` / `[/INST]` envelope, with system
//!   messages wrapped in `<<SYS>>` / `<</SYS>>`.
//! - **Plain**: simple `role: content` newline-separated format used
//!   as a fallback when the architecture is unknown.
//!
//! Architecture → template mapping (Qwen3 → ChatML, Llama → Llama3,
//! Mistral → MistralInst) is locked in by the three `test_template_for_*`
//! tests.
//!
//! Note: `[INST]` and `<<SYS>>` are literal Mistral template
//! sentinels — test fixtures only, not user-supplied content.
use super::*;
use crate::openai::types::ChatMessage;

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
        name: None,
    }
}

#[test]
fn test_chatml_user_only() {
    let prompt = build_prompt(ChatTemplate::ChatMl, &[msg("user", "Hello")]);
    assert_eq!(
        prompt,
        "<|endoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn test_llama3_user_only() {
    let prompt = build_prompt(ChatTemplate::Llama3, &[msg("user", "Hello")]);
    assert!(prompt.starts_with("<|begin_of_text|>"));
    assert!(prompt.contains("Hello"));
    assert!(prompt.ends_with("assistant<|end_header_id|>\n"));
}

#[test]
fn test_template_for_qwen3() {
    assert_eq!(
        ChatTemplate::for_architecture(Architecture::Qwen3),
        ChatTemplate::ChatMl
    );
}

#[test]
fn test_template_for_llama() {
    assert_eq!(
        ChatTemplate::for_architecture(Architecture::Llama),
        ChatTemplate::Llama3
    );
}

#[test]
fn test_template_for_mistral() {
    assert_eq!(
        ChatTemplate::for_architecture(Architecture::Mistral),
        ChatTemplate::MistralInst
    );
}

#[test]
fn test_llama3_system_and_user() {
    let prompt = build_prompt(
        ChatTemplate::Llama3,
        &[msg("system", "Be concise"), msg("user", "Hi")],
    );
    assert!(prompt.contains("system"));
    assert!(prompt.contains("Be concise"));
    assert!(prompt.contains("user"));
    assert!(prompt.contains("Hi"));
    assert!(prompt.ends_with("assistant<|end_header_id|>\n"));
}

#[test]
fn test_mistral_system_and_user() {
    let prompt = build_prompt(
        ChatTemplate::MistralInst,
        &[msg("system", "You are helpful"), msg("user", "Hello")],
    );
    assert_eq!(
        prompt,
        "<s>[INST] <<SYS>>\nYou are helpful\n<</SYS>>\n\nHello [/INST]"
    );
}

#[test]
fn test_mistral_multi_turn() {
    let prompt = build_prompt(
        ChatTemplate::MistralInst,
        &[
            msg("user", "Hi"),
            msg("assistant", "Hello!"),
            msg("user", "Bye"),
        ],
    );
    assert_eq!(prompt, "<s>[INST] Hi [/INST]Hello! [INST] Bye [/INST]");
}

#[test]
fn test_plain_prompt() {
    let prompt = build_prompt(ChatTemplate::Plain, &[msg("user", "Hi")]);
    assert_eq!(prompt, "user: Hi\n\nassistant: ");
}

#[test]
fn test_different_architectures_produce_different_prompts() {
    let messages = vec![msg("user", "Hello")];
    let qwen = build_prompt(ChatTemplate::ChatMl, &messages);
    let llama = build_prompt(ChatTemplate::Llama3, &messages);
    let mistral = build_prompt(ChatTemplate::MistralInst, &messages);
    assert_ne!(qwen, llama);
    assert_ne!(qwen, mistral);
    assert_ne!(llama, mistral);
}
