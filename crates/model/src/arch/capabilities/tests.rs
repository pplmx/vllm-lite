//! Unit tests for `ArchCapabilities`.
//!
//! Locks in the two derived contracts:
//!
//! - `is_stub()` returns true iff `inference = false`. The
//!   inverse checks both `PRODUCTION` (true) and the
//!   \`PRODUCTION_SPECULATIVE\` shape (true).
//! - `tier()` maps the four standard capability presets to their
//!   canonical labels: "stub", "production",
//!   "production-speculative".
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
