#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::from_slice;
use vllm_server::openai::types::ChatRequest;

// Fuzz `ChatRequest` JSON deserialization with arbitrary bytes.
//
// Goal: catch panics in serde_json -> ChatRequest deserialization.
// ChatRequest has 10+ fields including nested `ChatMessage`,
// `ResponseFormat`, sampling params, and tool-calling fields (when
// present). Adversarial inputs (deeply nested, recursive references,
// gigantic arrays) can stack-overflow or OOM some parsers.
//
// Real-world relevance: the OpenAI-compatible HTTP endpoint accepts
// untrusted JSON from network callers. A panic in deserialization
// is a remote DoS vector.

fuzz_target!(|data: &[u8]| {
    // Limit input size to avoid OOM from malicious huge inputs.
    // Most ChatRequest instances are well under 10KB; 1MB is a generous
    // cap that still allows fuzzing all reasonable structures.
    if data.len() > 1_000_000 {
        return;
    }
    // `Result::Err` is expected for malformed input; panic = bug.
    let _ = from_slice::<ChatRequest>(data);
});
