#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::from_slice;
use vllm_model::qwen3::Qwen3Config;

// Fuzz `Qwen3Config` JSON deserialization with arbitrary bytes.
//
// Goal: catch panics, infinite loops, or memory exhaustion in the
// serde_json -> Qwen3Config deserialization path triggered by malformed
// or adversarial HF config.json input. Real-world relevance: malformed
// HuggingFace model config files uploaded by users.
//
// Qwen3Config has 22+ nested fields including nested `TextConfig`,
// `RopeScaling`, and `RopeParameters` enums. The struct is permissive
// (all fields `Option<T>` with `#[serde(default)]`), so deserialization
// should never panic on missing/malformed fields — only on truly broken
// JSON syntax or extreme inputs.
fuzz_target!(|data: &[u8]| {
    // Attempt parse; ignore Err but panic = bug.
    let _ = from_slice::<Qwen3Config>(data);
});
