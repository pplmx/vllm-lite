//! Architecture-specific chat prompt formatting for the `OpenAI` chat API.

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
    #[must_use]
    pub const fn for_architecture(arch: Architecture) -> Self {
        match arch {
            Architecture::Qwen3 | Architecture::Qwen35 => Self::ChatMl,
            Architecture::Llama | Architecture::Gemma4 | Architecture::Mixtral => Self::Llama3,
            Architecture::Mistral => Self::MistralInst,
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
                    prompt.push_str(&format!(
                        "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                        system, msg.content
                    ));
                } else {
                    prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
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
}
