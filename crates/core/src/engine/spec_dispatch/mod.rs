//! Speculative decoding dispatch — the engine's speculative step path.
//!
//! This sub-tree implements [`Engine::step_speculative_inner`](super::Engine::step_speculative_inner)
//! and its supporting helpers:
//!
//! - [`warmup`] — KV cache warmup after prefill
//! - [`dispatch`] — `step_speculative_inner` (the top-level speculative step)
//! - [`drafts`] — draft generation (per-seq via resolver, or batched legacy)
//! - [`verify`] — logit-based verification with rejection sampling
//!
//! All methods are `impl crate::engine::Engine { ... }` blocks — split across files
//! to keep each file focused on a single concern. The split is purely
//! organizational; no behavior changed versus the original single-file
//! implementation.

#![allow(clippy::type_complexity, clippy::iter_cloned_collect)]

mod dispatch;
mod drafts;
mod verify;
mod warmup;

#[cfg(test)]
mod tests;
