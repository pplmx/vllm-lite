#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_saphyr::from_str;
use vllm_server::config::AppConfig;

// Fuzz `AppConfig` YAML deserialization with arbitrary bytes.
//
// Goal: catch panics, infinite loops, or memory exhaustion in the
// serde_saphyr -> AppConfig deserialization path triggered by malformed
// or adversarial YAML input.
//
// We ignore parse errors (they're expected) but any panic / OOM / hang
// in the deserializer is a real bug.
fuzz_target!(|data: &[u8]| {
    // Convert bytes to UTF-8 string (lossy is fine - we're fuzzing the parser).
    let yaml = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return, // Not valid UTF-8; can't fuzz YAML with it.
    };

    // Attempt parse; ignore Err but panic = bug.
    let _ = from_str::<AppConfig>(yaml);
});
