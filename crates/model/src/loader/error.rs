//! Error types for the model loader.
//!
//! Replaces `candle_core::Error::msg(...)` for typed failure modes like stub
//! architecture rejection. Per AGENTS.md "Error Type Conventions":
//! - `#[derive(thiserror::Error)]`
//! - `#[error("...")]` per variant
//! - `#[source]` for wrapped errors
//! - Typed errors in public APIs (no `Box<dyn Error>`)

/// `LoadError` — errors emitted by `ModelLoader`.
///
/// Currently scoped to typed stub-architecture rejection (CODE-04). Future
/// variants can be added as typed counterparts replace existing
/// `candle_core::Error::msg(...)` call sites.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Loading was attempted for a stub architecture without the
    /// `allow_stub` capability gate. Stubs do not perform real inference.
    #[error(
        "architecture '{name}' is a stub (tier: {tier}) and cannot be used for inference; \
         pass --allow-stub to override"
    )]
    StubNotAllowed { name: String, tier: String },
}
