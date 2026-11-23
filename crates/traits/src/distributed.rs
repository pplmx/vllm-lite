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

    /// The mixer must AVALANCHE: flipping one input bit must change many
    /// output bits.
    ///
    /// `cargo mutants` relaxed each `^=` in `xorshift_round` to `|=` / `&=`
    /// and each shift direction, and all six survived. They are not cosmetic:
    /// `|=` and `&=` are monotone in a way XOR is not, so a single-bit change
    /// in the parent can leave the hash unchanged, and the mask collapses the
    /// output space. Measured on this fixture, the worst-case single-bit flip
    /// changes 0 output bits under `|=`/`&=` and 6 under the real mixer, while
    /// a mid-size range keeps only 588 (of 1024) or 47 distinct hashes instead
    /// of 1024.
    ///
    /// A chain hash that collides does not just lose a cache hit: two
    /// different KV states become the same block, and the peer serves the
    /// WRONG KV for a prefix — silently, with no error anywhere.
    #[test]
    fn xorshift_round_avalanches_on_a_single_bit_flip() {
        let h = XorShiftHasher;
        let base_parent = 0x1234_5678_9ABC_DEF0_u64;
        let tokens: [TokenId; 3] = [1, 2, 3];
        let reference = h.hash_block(base_parent, &tokens);

        let mut worst = 64;
        for bit in 0..64 {
            let flipped = h.hash_block(base_parent ^ (1 << bit), &tokens);
            let changed = (reference ^ flipped).count_ones();
            assert_ne!(
                changed, 0,
                "flipping parent bit {bit} must change the hash (a constant \
                 hash makes every prefix collide)"
            );
            worst = worst.min(changed);
        }
        assert!(
            worst >= 4,
            "the weakest single-bit flip changed only {worst} of 64 output \
             bits; `|=`/`&=` collapse this to 0"
        );

        // The same must hold for the token dimension.
        let token_ref = h.hash_block(7, &[0xDEAD_BEEF]);
        let mut token_worst = 64;
        for bit in 0..32 {
            let flipped = h.hash_block(7, &[0xDEAD_BEEF ^ (1 << bit)]);
            token_worst = token_worst.min((token_ref ^ flipped).count_ones());
        }
        assert!(
            token_worst >= 4,
            "the weakest token bit-flip changed only {token_worst} of 64 bits"
        );
    }

    /// The mixer must keep the output space large, not merely distinct for the
    /// fixture in `xorshift_hasher_distributes_well`.
    ///
    /// That test asserts `>= 1000` unique out of 1024 for a 3-token fixture,
    /// which every `^=`→`|=`/`&=` mutant still satisfies — 1024 unique for all
    /// of them, because three rounds of `|=` on a seeded start still separate
    /// those particular inputs. A single-token range exposes the collapse.
    #[test]
    fn xorshift_round_does_not_collapse_the_output_space() {
        let h = XorShiftHasher;
        let unique: std::collections::HashSet<u64> =
            (0..8192u32).map(|i| h.hash_block(0, &[i])).collect();
        assert_eq!(
            unique.len(),
            8192,
            "8192 distinct single-token inputs must hash to 8192 distinct \
             values; `&=` drops this to 111 and `|=` to 3029"
        );
    }

    /// `hash_allocated_block`'s default impl must delegate to `hash_block`.
    ///
    /// The trait default ignores `block_id` on purpose (callers fold it in),
    /// and `cargo mutants` replaced it with the constants 0 and 1. Either
    /// constant makes every allocated block hash to the same value, so the
    /// per-block chain cursor degenerates and the first block id decides the
    /// whole chain. The fixture uses a hasher that does NOT override it.
    #[test]
    fn default_hash_allocated_block_matches_hash_block() {
        #[derive(Debug)]
        struct PlainHasher;
        impl BlockHasher for PlainHasher {
            fn hash_block(&self, parent_hash: u64, tokens: &[TokenId]) -> u64 {
                // A trivially observable mix, so a constant return is obvious.
                parent_hash
                    .wrapping_mul(31)
                    .wrapping_add(tokens.iter().map(|&t| u64::from(t)).sum())
            }
            fn name(&self) -> &'static str {
                "plain"
            }
        }

        let hasher = PlainHasher;
        let tokens: [TokenId; 2] = [4, 5];
        assert_eq!(
            hasher.hash_allocated_block(0, 11, &tokens),
            hasher.hash_block(11, &tokens),
            "the default impl must delegate, ignoring the block id"
        );
        assert_eq!(
            hasher.hash_allocated_block(999, 11, &tokens),
            hasher.hash_block(11, &tokens),
            "the block id is deliberately NOT mixed in by the default impl"
        );
        assert_ne!(hasher.hash_block(11, &tokens), 0);
        assert_ne!(hasher.hash_block(11, &tokens), 1);
    }

    /// The chain hash must NOT drift: these are the exact values the current
    /// mixer produces.
    ///
    /// `cargo mutants` flipped each shift direction in `xorshift_round`
    /// (`<< 13` -> `>> 13`, `<< 17` -> `>> 17`) and the three survivors in this
    /// file were exactly those: both reversed-shift variants still avalanche
    /// (min 14 and 12 changed bits) and still separate 8192 single-token inputs,
    /// so no property test distinguishes them — but they compute a DIFFERENT
    /// chain hash than the running cluster does.
    ///
    /// That is the one thing a distributed prefix cache cannot tolerate.
    /// `hash_block` is content-addressed across processes: node A hashes a
    /// block, node B looks it up by the hash it computes itself. If the two
    /// disagree — different binary, different mixer, a "harmless" refactor —
    /// every lookup misses and the cache silently degrades to zero hit rate,
    /// with no error anywhere. A golden vector turns that into a test failure.
    ///
    /// If a mixer change is intentional, update these constants in the SAME
    /// commit: the numbers are the wire contract.
    #[test]
    fn xorshift_hash_values_are_pinned_as_a_wire_contract() {
        let h = XorShiftHasher;
        assert_eq!(
            h.hash_block(0, &[]),
            11_400_714_819_323_198_485,
            "empty-token-parent-0 hash (the golden-ratio seed)"
        );
        assert_eq!(
            h.hash_block(0, &[1, 2, 3]),
            1_614_893_259_095_336_443,
            "hash_block(0, [1,2,3])"
        );
        assert_eq!(
            h.hash_block(42, &[7]),
            3_036_554_837_440_476_573,
            "hash_block(42, [7])"
        );
        assert_eq!(
            h.hash_allocated_block(1, 0, &[1, 2, 3]),
            15_187_365_440_758_544_606,
            "hash_allocated_block folds the block id in BEFORE the token loop"
        );
        assert_eq!(
            h.hash_allocated_block(42, 100, &[5, 6, 7, 8]),
            11_764_383_310_723_063_210,
            "hash_allocated_block(42, 100, [5,6,7,8])"
        );
    }

    /// A three-block chain must reproduce across calls and across hasher
    /// instances — the property the distributed cache relies on.
    #[test]
    fn xorshift_chain_is_reproducible_across_instances() {
        let chunks: [&[TokenId]; 3] = [&[1, 2, 3], &[4, 5, 6], &[7, 8]];

        let walk = |hasher: XorShiftHasher| {
            let mut parent = 0_u64;
            for (block_id, chunk) in chunks.iter().enumerate() {
                parent = hasher.hash_allocated_block(block_id, parent, chunk);
            }
            parent
        };

        let first = walk(XorShiftHasher);
        assert_eq!(first, walk(XorShiftHasher), "the same chain twice");
        assert_eq!(
            first,
            walk(XorShiftHasher::default()),
            "a fresh instance must produce the same chain"
        );
        assert_eq!(
            first, 8_738_591_894_055_008_282,
            "the chain's value is part of the same wire contract"
        );

        // A different parent must produce a different chain: the whole point
        // of chaining is that the hash covers the history.
        let other = {
            let mut parent = 1_u64;
            for (block_id, chunk) in chunks.iter().enumerate() {
                parent = XorShiftHasher.hash_allocated_block(block_id, parent, chunk);
            }
            parent
        };
        assert_ne!(other, first, "a different seed must change the chain");
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
