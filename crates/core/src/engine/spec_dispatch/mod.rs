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
//!
//! ## Refactor history (OPS-01 / v22.0)
//!
//! Prior to v20.0 the speculative step lived in a single
//! `crates/core/src/engine/speculative.rs` file (~900 LOC). During the
//! v20.0 module-tree restoration (Phase 26) and v20.1 cleanup, that
//! file was deleted and its contents split into this `spec_dispatch/`
//! sub-tree for single-responsibility / 300-LOC-per-file hygiene.
//!
//! OPS-01 ("speculative.rs mock usage fate") was originally a
//! v18.0-era audit finding asking whether the speculative path was
//! wired to a real draft loader or stub-only. That question is
//! **resolved** by the v18.0 introduction of
//! `DraftResolver::install_default_resolver` and the
//! `ServerDraftLoader` server wiring (see `crates/server/src/draft_loader.rs`);
//! the `Engine::step_speculative_inner` path now resolves drafts via
//! the configured `DraftResolver` and falls back to self-spec via
//! FALL-01 when no external draft is registered. There are no
//! speculative-only mock backends remaining in this module.

#![allow(clippy::type_complexity, clippy::iter_cloned_collect)]

mod dispatch;
mod drafts;
mod verify;
mod warmup;

#[cfg(test)]
mod tests;
