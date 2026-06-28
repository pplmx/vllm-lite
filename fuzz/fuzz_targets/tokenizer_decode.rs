#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz `tiktoken` decode with arbitrary token IDs.
//
// Goal: catch panics or unexpected behavior when the decoder encounters
// invalid token IDs (e.g., from corrupted token files, version mismatch,
// adversarial inputs from untrusted sources).
//
// tiktoken uses a BPE encoding with a ~100k entry cl100k vocabulary.
// Token IDs outside [0, vocab_size) should error gracefully or produce
// replacement bytes — they should never panic or trigger UB.
//
// Init: `tiktoken::get_encoding("cl100k_base")` returns a `&'static CoreBpe`
// cached internally via OnceLock, so the BPE rank table loads once.

fuzz_target!(|data: &[u8]| {
    // Skip inputs that are too small to form a token ID.
    if data.len() < 4 {
        return;
    }
    // Read up to 16 token IDs from the input as little-endian u32.
    let mut ids = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4).take(16) {
        let id = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        ids.push(id);
    }

    // `get_encoding` lazily initializes; it should never return None for
    // a well-known name, but bail if the crate fails to load.
    let Some(encoder) = tiktoken::get_encoding("cl100k_base") else {
        return;
    };

    // Decode must not panic on out-of-range token IDs. Result type is
    // Vec<u8> (raw bytes); UTF-8 decoding is the caller's responsibility
    // and may legitimately fail (handled via decode_to_string).
    let bytes = encoder.decode(&ids);
    // Sanity: result length is bounded by token count * max bytes/token.
    // We don't assert a tight bound — just that decode returned control.
    let _len = bytes.len();

    // Also exercise the UTF-8 path; an invalid UTF-8 sequence must return
    // Err rather than panic.
    let _text_result = encoder.decode_to_string(&ids);
});
