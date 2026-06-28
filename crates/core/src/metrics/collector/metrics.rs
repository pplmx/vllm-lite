// crates/core/src/metrics/collector/metrics.rs
//
// Metric type definitions used by `EnhancedMetricsCollector` and external
// exporters.

use std::fmt;

/// Kind of draft resolution outcome recorded by the metrics collector.
///
/// Each variant corresponds to a discrete counter:
/// - [`Self::External`] → `draft_resolutions_external_total`
///   (request had a `draft_model_id` that resolved to a loaded draft backend).
/// - [`Self::SelfSpec`] → `draft_resolutions_self_spec_total`
///   (fallback to the self-speculative path).
/// - [`Self::None`] → `draft_resolutions_none_total`
///   (no draft at all — pure target decode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DraftResolutionKind {
    /// Draft resolved from the registry (external model).
    External,
    /// Fallback to self-speculative decoding.
    SelfSpec,
    /// No speculative decoding — pure target decode.
    None,
}

impl DraftResolutionKind {
    /// Parse from a string. Accepts canonical names and common aliases
    /// (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "external" => Some(Self::External),
            "self_spec" | "self-spec" | "selfspec" => Some(Self::SelfSpec),
            "none" => Some(Self::None),
            _ => None,
        }
    }

    /// Canonical string representation (matches the historical wire values).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::External => "external",
            Self::SelfSpec => "self_spec",
            Self::None => "none",
        }
    }
}

impl fmt::Display for DraftResolutionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Snapshot of v18.0 multi-model speculative decoding metrics.
#[derive(Debug, Clone, Default)]
pub struct DraftMetricsSnapshot {
    pub resolutions_external_total: u64,
    pub resolutions_self_spec_total: u64,
    pub resolutions_none_total: u64,
    pub load_failures_total: u64,
    pub runtime_errors_total: u64,
}
