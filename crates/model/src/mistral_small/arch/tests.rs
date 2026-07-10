//! Unit tests for `MistralSmallArchitecture`.
//!
//! Covers the architecture-detection contract for the Mistral Small
//! (Mistral-Small-Instruct-2407 and similar) variants:
//!
//! 1. **Detection (positive)**: detects `model_type` in
//!    `{"mistral-small", "Mistral-Small-Instruct-2407"}`. The
//!    config must include `num_local_experts` for detection to
//!    fire (the field is part of the heuristic).
//! 2. **Detection (negative)**: does not detect `mistral`,
//!    `mistral-7b`, or `llama` (these belong to other
//!    architectures, not Mistral Small).
//! 3. **Name**: `name()` returns `"mistral-small"`.
//! 4. **Expert config**: `with_experts(num_experts, num_active)`
//!    records both values.
use super::*;
use serde_json::json;

#[test]
fn test_mistral_small_architecture_detect() {
    let arch = MistralSmallArchitecture::new();
    for model_type in ["mistral-small", "Mistral-Small-Instruct-2407"] {
        let config = json!({
            "model_type": model_type,
            "hidden_size": 4096,
            "num_local_experts": 8
        });
        assert!(
            arch.detect(&config),
            "Failed to detect model_type: {model_type}"
        );
    }
}

#[test]
fn test_mistral_small_architecture_not_detect_others() {
    let arch = MistralSmallArchitecture::new();
    for model_type in ["mistral", "mistral-7b", "llama"] {
        let config = json!({
            "model_type": model_type,
            "hidden_size": 4096
        });
        assert!(
            !arch.detect(&config),
            "Should not detect model_type: {model_type}"
        );
    }
}

#[test]
fn test_mistral_small_architecture_name() {
    let arch = MistralSmallArchitecture::new();
    assert_eq!(arch.name(), "mistral-small");
}

#[test]
fn test_mistral_small_expert_config() {
    let arch = MistralSmallArchitecture::with_experts(16, 4);
    assert_eq!(arch.num_experts, 16);
    assert_eq!(arch.num_active_experts, 4);
}
