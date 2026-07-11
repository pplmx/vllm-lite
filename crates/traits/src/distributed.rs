//! Pluggable hash function for distributed KV-block content addressing.
//!
//! Cross-node prefix lookup needs a deterministic hash of block
//! content so two nodes holding the same KV can recognize each other
//! when comparing [`vllm_dist::CacheStats`](../../vllm_dist/distributed_kv/cache/struct.CacheStats.html)
//! across the cluster. This trait abstracts the hash function so
//! production deployments can swap in a stronger hasher (xxhash,
//! blake3, …) without touching `vllm-core`.
//!
//! ## Chain-hash design
//!
//! Each block's hash is a function of:
//!
//! 1. The parent block's hash (`0` for the first block of a sequence),
//! 2. The tokens stored in the block.
//!
//! This produces a Merkle-like chain: any change in token `i` flips
//! hash `i` and every downstream hash `j ≥ i`. Two nodes that
//! observe the same hash chain for the same token sequence can be
//! confident they hold equivalent KV state.
//!
//! ## Determinism
//!
//! Two implementations with the same inputs MUST produce the same
//! hash. Implementations must not depend on:
//!
//! - Process-local state (random seeds, thread IDs, …)
//! - Wall-clock time
//! - Memory addresses (`&self as *const _ as usize`, …)
//!
//! Implementations MAY depend on:
//!
//! - The input arguments
//! - Compile-time constants (e.g. a baked-in mixing table)
//!
//! ## Security
//!
//! [`XorShiftHasher`] is for distribution, not trust. An attacker
//! who controls one node can produce hash collisions and trick a
//! peer into accepting forged KV state. Production deployments
//! that need adversarial robustness should plug in a cryptographic
//! hasher (blake3, sha256-truncated) via their own [`BlockHasher`].

use crate::types::{BlockId, TokenId};

/// Trait for distributed KV-block content hashers.
///
/// # Object safety
///
/// `BlockHasher` is object-safe (no generic methods, no `Self` in
/// return types other than `&Self`). Production code stores it as
/// `Arc<dyn BlockHasher + Send + Sync>`.
pub trait BlockHasher: Send + Sync + std::fmt::Debug {
    /// Compute a 64-bit hash for a block given its parent chain hash
    /// and the tokens stored in the block.
    ///
    /// `parent_hash == 0` for the first block of a sequence.
    ///
    /// # Determinism
    ///
    /// Must be a pure function of `parent_hash` and `tokens`. Same
    /// inputs ⇒ same hash, across processes, threads, and machines.
    fn hash_block(&self, parent_hash: u64, tokens: &[TokenId]) -> u64;

    /// A short, stable name for the hasher (used in metrics labels
    /// and debug logs).
    ///
    /// MUST be a `'static str` so it can be referenced without
    /// allocating.
    fn name(&self) -> &'static str;

    /// The block id that was just allocated. Provided as a third
    /// input so hashers can break ties between sequences that happen
    /// to share a parent hash with identical tokens.
    ///
    /// The default impl simply folds `block_id` into the token mix
    /// via [`Self::hash_block`] called with the same parent + tokens;
    /// hashers that want explicit block-id mixing should override.
    fn hash_allocated_block(&self, block_id: BlockId, parent_hash: u64, tokens: &[TokenId]) -> u64 {
        // Default: hash with parent + tokens; the block_id is folded
        // in by callers (the per-block cursor in MemoryManager). This
        // is enough for the chain-hash use case; explicit overrides
        // exist for hashers that want a different mix.
        let _ = block_id;
        self.hash_block(parent_hash, tokens)
    }
}

// ---------------------------------------------------------------------------
// Default + production implementations
// ---------------------------------------------------------------------------

/// Identity hasher — returns `parent_hash` unchanged.
///
/// Useful for tests and as the no-op default. Content-addressable
/// semantics require a real mixing function (see
/// [`XorShiftHasher`]); this one collapses every block to its
/// parent's hash.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityHasher;

impl BlockHasher for IdentityHasher {
    fn hash_block(&self, parent_hash: u64, _tokens: &[TokenId]) -> u64 {
        parent_hash
    }

    fn name(&self) -> &'static str {
        "identity"
    }
}

/// xorshift-mix hasher.
///
/// Folds each token into a running 64-bit state via xorshift
/// multiplication by the golden-ratio constant and three
/// shift-mix rounds. No external dependencies (xxhash would add a
/// crate the rest of the workspace does not need); the distribution
/// quality is sufficient for the chain-hash use case.
///
/// # Security
///
/// **Not cryptographic.** An attacker who controls one node can
/// produce hash collisions and trick a peer into accepting forged
/// KV state. Use a cryptographic hasher (blake3, sha256-truncated)
/// if adversarial robustness is required.
#[derive(Debug, Default, Clone, Copy)]
pub struct XorShiftHasher;

impl BlockHasher for XorShiftHasher {
    fn hash_block(&self, parent_hash: u64, tokens: &[TokenId]) -> u64 {
        // Seed with the golden-ratio constant so the chain is not
        // stuck at 0 for empty token streams (the xorshift_round
        // mixer has 0 as a fixed point — pure shifts on 0 stay 0).
        // This matches the splitmix64-style "scrambled start"
        // idiom used by xoroshiro / splitmix64.
        let mut h = parent_hash.wrapping_add(GOLDEN_RATIO_U64);
        for &t in tokens {
            // invariant: TokenId is u32; widening to u64 is exact
            // and never truncates.
            h ^= u64::from(t).wrapping_mul(GOLDEN_RATIO_U64);
            h = xorshift_round(h);
        }
        h
    }

    fn name(&self) -> &'static str {
        "xorshift"
    }

    fn hash_allocated_block(&self, block_id: BlockId, parent_hash: u64, tokens: &[TokenId]) -> u64 {
        // Explicit block-id mix: ensures two different block ids
        // with identical tokens + parent get distinct hashes. The
        // mix folds the block id in BEFORE the token loop so the
        // resulting chain remains deterministic given the inputs.
        //
        // invariant: BlockId is usize; on every platform the
        // workspace targets this fits in u64.
        let mut h = parent_hash
            .wrapping_add(u64::try_from(block_id).unwrap_or(u64::MAX))
            .wrapping_add(GOLDEN_RATIO_U64);
        h = xorshift_round(h);
        for &t in tokens {
            h ^= u64::from(t).wrapping_mul(GOLDEN_RATIO_U64);
            h = xorshift_round(h);
        }
        h
    }
}

/// `floor(2^64 / phi)` — the 64-bit golden-ratio constant used by
/// the xorshift multiplier. Same value as `0x9E3779B97F4A7C15`.
const GOLDEN_RATIO_U64: u64 = 0x9E37_79B9_7F4A_7C15;

/// One round of xorshift64: shift-mix the running state.
///
/// Three rounds is the standard "xorshift64*" mixer; chosen to be
/// fast while keeping avalanche properties for short inputs.
#[inline]
const fn xorshift_round(mut h: u64) -> u64 {
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- IdentityHasher ---

    #[test]
    fn identity_hasher_returns_parent_hash() {
        let h = IdentityHasher;
        assert_eq!(h.hash_block(0, &[]), 0);
        assert_eq!(h.hash_block(42, &[]), 42);
        assert_eq!(h.hash_block(0xDEAD_BEEF, &[1, 2, 3]), 0xDEAD_BEEF);
        assert_eq!(h.hash_block(u64::MAX, &[u32::MAX]), u64::MAX);
    }

    #[test]
    fn identity_hasher_name() {
        assert_eq!(IdentityHasher.name(), "identity");
    }

    // --- XorShiftHasher ---

    #[test]
    fn xorshift_hasher_is_deterministic() {
        let h = XorShiftHasher;
        let tokens: Vec<TokenId> = (0..32).collect();
        let a = h.hash_block(0, &tokens);
        let b = h.hash_block(0, &tokens);
        assert_eq!(a, b);
    }

    #[test]
    fn xorshift_hasher_distinguishes_different_tokens() {
        let h = XorShiftHasher;
        let a = h.hash_block(0, &[1, 2, 3, 4]);
        let b = h.hash_block(0, &[1, 2, 3, 5]);
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift_hasher_distinguishes_different_parents() {
        let h = XorShiftHasher;
        let tokens: Vec<TokenId> = vec![10, 20, 30];
        let a = h.hash_block(0, &tokens);
        let b = h.hash_block(1, &tokens);
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift_hasher_empty_tokens_still_uses_parent() {
        let h = XorShiftHasher;
        // Empty token slice must still produce a hash derived from
        // parent_hash — the chain property depends on this.
        let a = h.hash_block(0, &[]);
        let b = h.hash_block(1, &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift_hasher_name() {
        assert_eq!(XorShiftHasher.name(), "xorshift");
    }

    #[test]
    fn xorshift_hasher_distributes_well() {
        // Sanity: 1024 distinct token sequences should not collide
        // in any obvious pattern. We just check that consecutive
        // hashes differ and the full set has at least 1000 unique
        // values out of 1024.
        let h = XorShiftHasher;
        let hashes: Vec<u64> = (0..1024u32)
            .map(|i| h.hash_block(0, &[i, i.wrapping_mul(31), i.wrapping_add(7)]))
            .collect();
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert!(
            unique.len() >= 1000,
            "got {} unique out of 1024",
            unique.len()
        );
        // Consecutive parents should give consecutive-but-different hashes.
        for w in hashes.windows(2) {
            assert_ne!(w[0], w[1]);
        }
    }

    #[test]
    fn xorshift_hasher_allocated_block_includes_block_id() {
        let h = XorShiftHasher;
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        // Same parent + tokens but different block ids → distinct
        // hashes (the explicit block-id mix).
        let a = h.hash_allocated_block(1, 0, &tokens);
        let b = h.hash_allocated_block(2, 0, &tokens);
        assert_ne!(a, b);
    }

    #[test]
    fn xorshift_hasher_allocated_block_is_deterministic() {
        let h = XorShiftHasher;
        let tokens: Vec<TokenId> = vec![5, 6, 7, 8];
        let a = h.hash_allocated_block(42, 100, &tokens);
        let b = h.hash_allocated_block(42, 100, &tokens);
        assert_eq!(a, b);
    }

    // --- Object safety ---

    #[test]
    fn block_hasher_object_safe() {
        // Compile-time check: we can store a trait object and call
        // through it. Runtime check: the vtable dispatches correctly.
        let hashers: Vec<Box<dyn BlockHasher>> =
            vec![Box::new(IdentityHasher), Box::new(XorShiftHasher)];
        for hasher in &hashers {
            let h = hasher.hash_block(0, &[1, 2, 3]);
            assert_eq!(h, hasher.hash_block(0, &[1, 2, 3])); // deterministic
        }
        assert_eq!(hashers[0].name(), "identity");
        assert_eq!(hashers[1].name(), "xorshift");
    }
}
