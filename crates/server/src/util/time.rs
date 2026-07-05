//! Time utilities for the server crate.
//!
//! Centralises the `"seconds since UNIX_EPOCH as i64"` idiom so we don't
//! repeat the `SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| ...)`
//! dance at every call site. Also surfaces a `now_millis` variant for
//! callers that want ms-precision timestamps.
//!
//! # Why a helper?
//!
//! Naïve `SystemTime::now().duration_since(UNIX_EPOCH).unwrap()` panics on
//! systems where the wall clock has been adjusted backwards across
//! `UNIX_EPOCH` (rare but observed on NTP-misconfigured hosts and VMs after
//! suspend/resume). The helper here silently saturates to 0 / `i64::MAX`
//! — never panics — and keeps the existing batch-handler behaviour
//! (where the previous code already used `map_or(0, ...)`).
//!
//! # When to use `i64` vs `u64`
//!
//! OpenAI wire format requires `i64` (it serialises as a signed integer).
//! Internal metrics may prefer `u64` (cheaper, no sign bit). Pick the variant
//! that matches the wire format of the receiving system.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current wall-clock time as **seconds since `UNIX_EPOCH`**,
/// saturated to `0` if the system clock is set before `UNIX_EPOCH`.
///
/// Returns `i64::MAX` if the value overflows (year ~292 billion AD). This
/// matches the OpenAI wire-format expectation.
#[must_use]
pub fn unix_now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// Returns the current wall-clock time as **milliseconds since `UNIX_EPOCH`**.
///
/// Saturates to `0` (clock before epoch) or `i64::MAX` (overflow). Useful for
/// distributed-cache TTLs and tracing span timestamps that need ms precision.
#[must_use]
pub fn unix_now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_now_secs_is_sensibly_after_2024() {
        // We just need a sanity check that we're not getting 1970.
        // 2024-01-01 UTC = 1_704_067_200.
        let now = unix_now_secs();
        assert!(now > 1_704_067_200, "expected post-2024 timestamp, got {now}");
    }

    #[test]
    fn unix_now_millis_is_consistent_with_secs() {
        let secs = unix_now_secs();
        let millis = unix_now_millis();
        // Allow ±2 seconds of skew across the two reads.
        let millis_as_secs = millis / 1000;
        assert!(
            (millis_as_secs - secs).abs() <= 2,
            "millis-derived seconds {millis_as_secs} drifted from secs {secs}"
        );
    }
}
