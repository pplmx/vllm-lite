// crates/server/src/bootstrap/mod.rs
//
// Server startup helpers split by concern:
// - `engine`    — engine construction (loader → model → optional draft →
//                 Engine) + speculative-decoding knob wiring
// - `tokenizer` — tokenizer loading from `<model_dir>/tokenizer.json` with
//                 graceful fallback to a default-constructed tokenizer
// - `handlers`  — `/health`, `/ready`, `/metrics` HTTP handlers used by
//                 Kubernetes liveness/readiness probes and Prometheus
//                 scrapers

pub mod engine;
pub mod handlers;
pub mod tokenizer;
