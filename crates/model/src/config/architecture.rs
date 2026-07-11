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

/// Configuration for `RoPE`. Constructed via the `builder()` associated function or by deserializing from JSON / TOML. Pass-by-value to construction APIs.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

/// Architecture: architecture enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Unknown / unrecognised model architecture. Use as a fallback when
    /// config parsing yields no recognised `model_type`; the corresponding
    /// trait impl is [`crate::arch::UnknownArchitecture`] (always errors
    /// on `create_block` / `create_model`).
    Unknown,
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

    /// Wire-format name (matches HF `model_type` for the supported
    /// architectures; "unknown" for [`Self::Unknown`]).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen3.5",
            Self::Gemma4 => "gemma4",
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
