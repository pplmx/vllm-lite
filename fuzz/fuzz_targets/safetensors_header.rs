#![no_main]

use libfuzzer_sys::fuzz_target;
use safetensors::SafeTensors;

// Fuzz safetensors header deserialization with arbitrary bytes.
//
// Goal: catch panics in the safetensors header parsing path
// (length-prefix validation, JSON header parse, tensor metadata
// validation). Real-world relevance: malformed checkpoint files
// uploaded by users must not crash the loader.
//
// We ignore `Err` (it's expected for malformed input) but any panic /
// OOM / UB in the deserializer is a real bug.
fuzz_target!(|data: &[u8]| {
    let _ = SafeTensors::deserialize(data);
});
