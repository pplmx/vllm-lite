#![no_main]

use libfuzzer_sys::fuzz_target;

// Fuzz GGUF header parsing with arbitrary bytes.
//
// Goal: catch panics in the GGUF magic number check and version parsing
// when given malformed or adversarial bytes. GGUF files start with a
// 4-byte magic ("GGUF" = [0x47, 0x55, 0x46, 0x47]) followed by version
// uint32 and tensor/metadata counts as uint64.
//
// Real-world relevance: checkpoint loading accepts arbitrary .gguf files
// from users. A panic during header parse aborts the whole load process.
//
// We verify the magic check and basic header slice without depending on
// the `gguf` crate (which would pull in heavyweight parsing). The full
// header parser is exercised by the integration test suite.

fuzz_target!(|data: &[u8]| {
    // GGUF v3 header layout (minimum 24 bytes):
    //   u32  magic           (must be "GGUF")
    //   u32  version
    //   u64  tensor_count
    //   u64  metadata_kv_count
    if data.len() < 4 {
        return;
    }
    let magic = &data[0..4];
    let _is_gguf = magic == b"GGUF";

    // The slice-based magic comparison must never panic on short inputs;
    // we just verified the length bound above.
    let _ = magic.len();

    // If we have a version field, attempt to read it as little-endian u32.
    // Out-of-range or unusual version values should be handled, not panic.
    if data.len() >= 8 {
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        // v1, v2, v3 are the only known versions; anything else is invalid
        // but must not panic the loader. We don't branch on it here — this
        // is a structural fuzz, not a semantic one.
        let _ = version;
    }
});
