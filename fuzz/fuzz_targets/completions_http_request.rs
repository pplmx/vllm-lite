#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::from_slice;
use vllm_server::openai::types::CompletionRequest;

// Fuzz `CompletionRequest` JSON deserialization with arbitrary bytes.
//
// Goal: catch panics in serde_json -> CompletionRequest deserialization.
// CompletionRequest has 10+ fields including prompt (String | Vec<String> |
// Vec<i64> | array-of-arrays), logit_bias (HashMap<TokenId, f32>), sampling
// params, and the `n`/`stop`/`best_of` fields that were added as v0.3 wire-type
// compliance targets. Adversarial inputs (deeply nested, recursive references,
// gigantic arrays) can stack-overflow or OOM some parsers.
//
// Real-world relevance: the /v1/completions endpoint accepts untrusted JSON
// from network callers. A panic in deserialization is a remote DoS vector.
// This target mirrors openai_http_request.rs which covers ChatRequest, so both
// endpoint entry points are fuzzed independently.

fuzz_target!(|data: &[u8]| {
    // Limit input size to avoid OOM from malicious huge inputs.
    // CompletionRequest prompts can be large (batch token arrays), but 10MB
    // is a sane DoS bound that still allows fuzzing all reasonable structures.
    if data.len() > 10_000_000 {
        return;
    }
    // `Result::Err` is expected for malformed input; panic = bug.
    let _ = from_slice::<CompletionRequest>(data);
});
