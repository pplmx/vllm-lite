//! Qwen3 `RoPE` configuration: theta, partial rotary factor, mrope sections for vision-language variants.
//!
//! Most variants use the standard `rope_theta=1_000_000`; vision-language
//! Qwen3 models add the `MRoPE` sections (temporal / height / width).

// crates/model/src/qwen3/config/rope.rs
//
// RoPE-related configuration types for Qwen3:
// `RopeType`, `RopeScaling`, `RopeParameters`.

use serde::Deserialize;

/// `RoPE` scaling type (HuggingFace-compatible).
///
/// Maps the `rope_type` string field found in `HuggingFace`
/// `RopeScaling` and `RopeParameters` JSON blobs to a typed enum.
/// Unknown values deserialize to [`RopeType::Other`] for graceful
/// forward compatibility with new HF rope types.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Deserialize, serde::Serialize,
)]
#[serde(rename_all = "lowercase")]
pub enum RopeType {
    /// Default standard `RoPE` without scaling (Qwen3 baseline).
    #[default]
    Default,
    /// Linear position interpolation.
    Linear,
    /// Dynamic NTK-aware scaling.
    Dynamic,
    /// `YaRN` (Yet another `RoPE` extensioN).
    Yarn,
    /// Su `RoPE` (`RoPE` in any precision).
    Su,
    /// Other / unknown rope type (serde fallback for forward compat).
    #[serde(other)]
    Other,
}

impl RopeType {
    /// Canonical string representation (matches the HF wire value).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Linear => "linear",
            Self::Dynamic => "dynamic",
            Self::Yarn => "yarn",
            Self::Su => "su",
            Self::Other => "other",
        }
    }

    /// Parse from a string (case-insensitive). Unknown values map to
    /// [`RopeType::Other`] rather than `None`, so callers can
    /// distinguish "missing" from "unknown".
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "linear" => Some(Self::Linear),
            "dynamic" => Some(Self::Dynamic),
            "yarn" => Some(Self::Yarn),
            "su" => Some(Self::Su),
            _ => Some(Self::Other),
        }
    }
}

impl std::fmt::Display for RopeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `RopeScaling`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScaling {
    /// Which `RoPE` scaling algorithm to use (None = default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_type: Option<RopeType>,
    /// Linear-interpolation / `YaRN` scaling factor.
    #[serde(default)]
    pub factor: Option<f32>,
    /// Original context length this scaling was tuned for.
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    /// Attention scaling factor applied alongside `RoPE` (`YaRN`).
    #[serde(default)]
    pub attn_factor: Option<f32>,
    /// Fraction of each head dimension that receives rotary embeddings (Qwen3 uses 0.25).
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// `MRoPE` axis section sizes (temporal / height / width).
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// Su RoPE per-dim factor for high-frequency dims (length head_dim/2).
    /// Used only when `rope_type == Su`; ignored otherwise.
    #[serde(default)]
    pub short_factor: Option<Vec<f32>>,
    /// Su RoPE per-dim factor for low-frequency dims (length head_dim/2).
    /// Used only when `rope_type == Su`; ignored otherwise.
    #[serde(default)]
    pub long_factor: Option<Vec<f32>>,
}

/// `RopeParameters`. See the type definition for fields and behavior.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeParameters {
    /// Which `RoPE` scaling algorithm to use (None = default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_type: Option<RopeType>,
    /// Base theta for the rotary inverse-frequency computation (Qwen3 = `1_000_000`).
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Fraction of each head dimension that receives rotary embeddings.
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    /// `MRoPE` axis section sizes (temporal / height / width).
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// Whether `MRoPE` axes are interleaved rather than split.
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_type_serde_lowercase() {
        // Known variants serialize as their lowercase HF wire name.
        assert_eq!(
            serde_json::to_string(&RopeType::Default).unwrap(),
            "\"default\""
        );
        assert_eq!(
            serde_json::to_string(&RopeType::Linear).unwrap(),
            "\"linear\""
        );
        assert_eq!(
            serde_json::to_string(&RopeType::Dynamic).unwrap(),
            "\"dynamic\""
        );
        assert_eq!(serde_json::to_string(&RopeType::Yarn).unwrap(), "\"yarn\"");
        assert_eq!(serde_json::to_string(&RopeType::Su).unwrap(), "\"su\"");

        // Known lowercase strings deserialize to the matching variant.
        let parsed: RopeType = serde_json::from_str("\"default\"").unwrap();
        assert_eq!(parsed, RopeType::Default);
        let parsed: RopeType = serde_json::from_str("\"linear\"").unwrap();
        assert_eq!(parsed, RopeType::Linear);
        let parsed: RopeType = serde_json::from_str("\"yarn\"").unwrap();
        assert_eq!(parsed, RopeType::Yarn);

        // Unknown values map to Other (graceful forward compat).
        let other: RopeType = serde_json::from_str("\"future_unknown\"").unwrap();
        assert_eq!(other, RopeType::Other);
    }

    #[test]
    fn rope_type_parse_roundtrip() {
        for kind in [
            RopeType::Default,
            RopeType::Linear,
            RopeType::Dynamic,
            RopeType::Yarn,
            RopeType::Su,
        ] {
            assert_eq!(RopeType::parse(kind.as_str()), Some(kind));
            assert_eq!(RopeType::parse(&kind.to_string()), Some(kind));
        }
    }

    #[test]
    fn rope_type_default_is_default_variant() {
        assert_eq!(RopeType::default(), RopeType::Default);
    }

    #[test]
    fn rope_type_optional_field_default_is_none() {
        let json = "{}";
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert!(parsed.rope_type.is_none());

        let parsed: RopeParameters = serde_json::from_str(json).unwrap();
        assert!(parsed.rope_type.is_none());
    }

    #[test]
    fn rope_type_optional_field_known_string_deserializes() {
        let json = r#"{"rope_type": "default", "factor": 2.0}"#;
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.rope_type, Some(RopeType::Default));
        assert_eq!(parsed.factor, Some(2.0));

        let json = r#"{"rope_type": "yarn"}"#;
        let parsed: RopeParameters = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.rope_type, Some(RopeType::Yarn));
    }

    // === Phase 16: Su RoPE config fields ===

    #[test]
    fn rope_scaling_short_factor_deserializes() {
        let json = r#"{"short_factor": [1.0, 1.5, 2.0, 2.5]}"#;
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.short_factor.as_deref(),
            Some([1.0_f32, 1.5, 2.0, 2.5].as_slice())
        );
    }

    #[test]
    fn rope_scaling_long_factor_deserializes() {
        let json = r#"{"long_factor": [4.0, 4.5, 5.0, 5.5]}"#;
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.long_factor.as_deref(),
            Some([4.0_f32, 4.5, 5.0, 5.5].as_slice())
        );
    }

    #[test]
    fn rope_scaling_missing_new_fields_defaults_to_none() {
        let json = r#"{"rope_type": "yarn", "factor": 4.0}"#;
        let parsed: RopeScaling = serde_json::from_str(json).unwrap();
        assert!(parsed.short_factor.is_none());
        assert!(parsed.long_factor.is_none());
    }
}
