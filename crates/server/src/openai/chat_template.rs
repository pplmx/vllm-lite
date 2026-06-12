//! Architecture-specific chat prompt formatting for the OpenAI chat API.

use super::types::ChatMessage;
use vllm_model::config::Architecture;

/// Prompt formatting strategy selected from the loaded model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// ChatML (`<|im_start|>` / ``) used by Qwen models.
    ChatMl,
    /// Llama 3 header tokens.
    Llama3,
    /// Mistral/Llama-2 instruct `[INST]` wrapping.
    MistralInst,
    /// Plain role-prefixed lines (fallback).
    Plain,
}

impl ChatTemplate {
    pub fn for_architecture(arch: Architecture) -> Self {
        match arch {
            Architecture::Qwen3 | Architecture::Qwen35 => Self::ChatMl,
            Architecture::Llama | Architecture::Gemma4 | Architecture::Mixtral => Self::Llama3,
            Architecture::Mistral => Self::MistralInst,
        }
    }
}

pub fn build_prompt(template: ChatTemplate, messages: &[ChatMessage]) -> String {
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

fn build_mistral_inst_prompt(messages: &[ChatMessage]) -> String {
    let system: String = messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let mut prompt = String::from("<s>");
    let mut expect_user = true;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {}
            "user" if expect_user => {
                if system.is_empty() {
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                } else {
                    prompt.push_str(&format!(
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                        system, msg.content
                    ));
                }
                expect_user = false;
            }
            "user" => {
                prompt.push_str(&format!(" [INST] {} [/INST]", msg.content));
                expect_user = false;
            }
            "assistant" => {
                prompt.push_str(&msg.content);
                expect_user = true;
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

#[cfg(test)]
mod tests {
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
}
