//! Architecture-specific chat prompt formatting for the `OpenAI` chat API.

use super::types::ChatMessage;
use std::fmt::Write;
use vllm_model::config::Architecture;

/// Prompt formatting strategy selected from the loaded model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `ChatML` (`<|im_start|>` / `<|im_end|>`) used by Qwen models.
    ChatMl,
    /// Llama 3 header tokens.
    Llama3,
    /// Mistral/Llama-2 instruct `[INST]` wrapping.
    MistralInst,
    /// Plain role-prefixed lines (fallback).
    Plain,
}

impl ChatTemplate {
    /// Select the appropriate chat template based on the model architecture.
    #[must_use]
    pub const fn for_architecture(arch: Architecture) -> Self {
        match arch {
            Architecture::Qwen3 | Architecture::Qwen35 => Self::ChatMl,
            Architecture::Llama | Architecture::Gemma4 | Architecture::Mixtral => Self::Llama3,
            Architecture::Mistral => Self::MistralInst,
            // Unknown / unrecognised architectures fall back to the plain
            // role-prefixed template.
            Architecture::Unknown => Self::Plain,
        }
    }
}

pub(crate) fn build_prompt(template: ChatTemplate, messages: &[ChatMessage]) -> String {
    match template {
        ChatTemplate::ChatMl => build_chatml_prompt(messages),
        ChatTemplate::Llama3 => build_llama3_prompt(messages),
        ChatTemplate::MistralInst => build_mistral_inst_prompt(messages),
        ChatTemplate::Plain => build_plain_prompt(messages),
    }
}

fn build_chatml_prompt(messages: &[ChatMessage]) -> String {
    let im_start = "<|im_start|>";
    let im_end = "<|im_end|>";
    let mut prompt = String::from("<|endoftext|>");

    for msg in messages {
        match msg.role.as_str() {
            "system" | "user" | "assistant" => {
                prompt.push_str(im_start);
                prompt.push_str(&msg.role);
                prompt.push('\n');
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

fn build_llama3_prompt(messages: &[ChatMessage]) -> String {
    let bos = "<|begin_of_text|>";
    let start_header = "<|start_header_id|>";
    let end_header = "<|end_header_id|>";
    let eot = "<|eot_id|>";

    let mut prompt = String::from(bos);

    for msg in messages {
        match msg.role.as_str() {
            "system" | "user" | "assistant" => {
                prompt.push_str(start_header);
                prompt.push_str(&msg.role);
                prompt.push_str(end_header);
                prompt.push('\n');
                prompt.push_str(&msg.content);
                prompt.push_str(eot);
            }
            _ => {}
        }
    }

    prompt.push_str(start_header);
    prompt.push_str("assistant");
    prompt.push_str(end_header);
    prompt.push('\n');
    prompt
}

#[allow(clippy::match_same_arms)]
fn build_mistral_inst_prompt(messages: &[ChatMessage]) -> String {
    let system: String = messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = String::from("<s>");
    let mut first_user = true;
    let mut after_assistant = false;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {}
            "user" => {
                if after_assistant {
                    prompt.push(' ');
                }
                if first_user && !system.is_empty() {
                    let _ = write!(
                        prompt,
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                        system, msg.content
                    );
                } else {
                    let _ = write!(prompt, "[INST] {} [/INST]", msg.content);
                }
                first_user = false;
                after_assistant = false;
            }
            "assistant" => {
                prompt.push_str(&msg.content);
                after_assistant = true;
            }
            _ => {}
        }
    }

    prompt
}

fn build_plain_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        if matches!(msg.role.as_str(), "system" | "user" | "assistant") {
            prompt.push_str(&msg.role);
            prompt.push_str(": ");
            prompt.push_str(&msg.content);
            prompt.push_str("\n\n");
        }
    }
    prompt.push_str("assistant: ");
    prompt
}

// Unit tests are extracted to `tests.rs` (sibling) to keep this
// template module under the 800-line soft cap. They cover the four
// built-in templates (ChatML / Llama3 / MistralInst / Plain) and
// the architecture → template mapping (Qwen3 → ChatML, Llama →
// Llama3, Mistral → MistralInst).
#[cfg(test)]
mod tests;
