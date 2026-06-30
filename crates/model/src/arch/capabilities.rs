#![allow(clippy::module_name_repetitions)]
//! Architecture capability flags for production vs stub models.

/// Capability flags describing what an architecture can do at load/inference time.
#[allow(clippy::struct_excessive_bools)] // intentional feature flags, not a state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchCapabilities {
    /// Real token generation (not stub / passthrough).
    pub inference: bool,
    /// Paged KV cache integration.
    pub paged_kv: bool,
    /// Loads and applies checkpoint weights.
    pub weight_load: bool,
    /// Supports `forward_to_layer` / speculative draft paths.
    pub speculative: bool,
}

impl ArchCapabilities {
    /// Full production causal LM (Llama, Mistral, Mixtral, Gemma4, …).
    pub const PRODUCTION: Self = Self {
        inference: true,
        paged_kv: true,
        weight_load: true,
        speculative: false,
    };

    /// Production LM with speculative-decoding hooks (Qwen3).
    pub const PRODUCTION_SPECULATIVE: Self = Self {
        inference: true,
        paged_kv: true,
        weight_load: true,
        speculative: true,
    };

    /// Hybrid production path without speculative hooks (reserved for partial integrations).
    pub const HYBRID: Self = Self {
        inference: true,
        paged_kv: true,
        weight_load: true,
        speculative: false,
    };

    /// Placeholder architecture — must not be used for real serving without opt-in.
    pub const STUB: Self = Self {
        inference: false,
        paged_kv: false,
        weight_load: false,
        speculative: false,
    };

    /// Returns true when the architecture is a stub (no real inference).
    #[must_use]
    pub const fn is_stub(self) -> bool {
        !self.inference
    }

    /// Human-readable maturity tier for logging.
    #[must_use]
    pub const fn tier(self) -> &'static str {
        if self.is_stub() {
            "stub"
        } else if self.speculative {
            "production-speculative"
        } else if self.inference && self.paged_kv {
            "production"
        } else {
            "partial"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_is_detected() {
        assert!(ArchCapabilities::STUB.is_stub());
        assert!(!ArchCapabilities::PRODUCTION.is_stub());
    }

    #[test]
    fn test_tier_labels() {
        assert_eq!(ArchCapabilities::STUB.tier(), "stub");
        assert_eq!(ArchCapabilities::PRODUCTION.tier(), "production");
        assert_eq!(
            ArchCapabilities::PRODUCTION_SPECULATIVE.tier(),
            "production-speculative"
        );
    }
}
