#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::from_slice;
use vllm_server::openai::batch::types::SimpleBatchRequest;

// Fuzz `SimpleBatchRequest` JSON deserialization with arbitrary bytes.
//
// Goal: catch panics in batch API request parsing. Batch endpoints accept
// multi-request payloads (each batch contains up to N inner requests);
// malformed JSON in batch input files is a common source of production
// panics.
//
// Real-world relevance: batch input files are uploaded by users and may
// contain millions of records; even small per-record panics become
// systemic under that load.

fuzz_target!(|data: &[u8]| {
    // Larger cap than ChatRequest because batch payloads may legitimately
    // contain many inner requests. 10MB is still a sane DoS bound.
    if data.len() > 10_000_000 {
        return;
    }
    let _ = from_slice::<SimpleBatchRequest>(data);
});
