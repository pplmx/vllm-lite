#![allow(clippy::module_name_repetitions)]
//! Tokenizer helpers for on-disk Qwen3 integration tests.

use vllm_model::tokenizer::Tokenizer;

use super::qwen3;

pub fn qwen3_tokenizer() -> Tokenizer {
    qwen3::tokenizer()
}

pub fn is_printable_text(text: &str) -> bool {
    !text.is_empty()
        && !text.chars().any(|c| c == '\u{FFFD}')
        && text.chars().any(char::is_alphabetic)
}
