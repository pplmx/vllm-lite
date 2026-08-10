//! Constant-time comparison primitives.
//!
//! API-key verification must not leak match position through wall-clock
//! time. A short-circuiting comparison (`Vec::contains` / `String ==
//! String`) returns as soon as the first differing byte is seen, so an
//! attacker iterating candidate keys observes a time that grows with the
//! number of correct prefix bytes — recovering the key prefix
//! byte-by-byte over the network (RIL ISS-049). `constant_time_eq`
//! always iterates every byte, so its runtime is independent of where
//! the first mismatch falls.

/// Constant-time bytewise string equality.
///
/// Leaks only the byte *length* (unavoidable without padding the
/// comparison to a fixed size — and both operands here are presented /
/// configured bearer tokens whose lengths are public data). The full
/// byte comparison always runs, so the number of matching prefix bytes
/// does not shorten the operation. A length mismatch returns `false`
/// immediately.
#[must_use]
pub fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.bytes().zip(b.bytes()) {
        // Accumulate rather than short-circuit: every byte must be
        // examined so the timing is independent of match position.
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::constant_time_eq;

    #[test]
    fn equal_strings_match() {
        assert!(constant_time_eq("sk-abc123", "sk-abc123"));
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn different_strings_do_not_match() {
        assert!(!constant_time_eq("sk-abc123", "sk-abc124"));
        assert!(!constant_time_eq("sk-abc123", "sk-xbc123"));
        assert!(!constant_time_eq("sk-abc123", "sk-abc1234"));
    }

    #[test]
    fn first_byte_difference_still_matches_the_full_sweep_semantics() {
        // The contract is equivalence, not internals — but pin that a
        // first-byte mismatch correctly yields false (the old bug was a
        // comparator that was *correct* yet *fast*: timing, not value).
        assert!(!constant_time_eq("a", "b"));
        assert!(!constant_time_eq("aaaa", "baaa"));
    }
}
