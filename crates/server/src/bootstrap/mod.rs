// crates/server/src/bootstrap/mod.rs
//
// Server startup helpers split by concern:
// - `engine`    — engine construction (loader → model → optional draft →
//                 Engine) + speculative-decoding knob wiring
// - `tokenizer` — tokenizer loading from `<model_dir>/tokenizer.json` with
//                 graceful fallback to a default-constructed tokenizer
//
// (The probe handlers — `/health`, `/ready`, `/metrics` — live in
// `crate::health_handlers` so integration tests can mount them against
// a controlled `ApiState`.)

pub mod engine;
pub mod tokenizer;
