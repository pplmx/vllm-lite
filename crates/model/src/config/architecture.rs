//! Model architecture types.
//!
//! Defines the Architecture enum and related types for supporting
//! multiple model architectures.

/// `LayerType`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

/// Configuration for RoPE. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

/// Architecture: architecture enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Qwen3,
    Qwen35,
    Llama,
    Mistral,
    Gemma4,
    Mixtral,
}

impl Architecture {
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "llama" => Some(Self::Llama),
            "mistral" => Some(Self::Mistral),
            "mixtral" => Some(Self::Mixtral),
            "qwen3" => Some(Self::Qwen3),
            "qwen3.5" | "qwen3_5" => Some(Self::Qwen35),
            "gemma4" => Some(Self::Gemma4),
            _ => None,
        }
    }
}

/// `AttentionType`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Mha,
    Gqa,
    SlidingWindow,
}

/// `NormType`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

/// `MlpType`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    SwiGLU,
    GatedMLP,
}
