//! KV cache utilities: block allocation and hash-based prefix cache.
//!
//! Production scheduling uses [`crate::scheduler::RadixTree`] for prefix matching.
//! This module re-exports the hash-based [`PrefixCache`] used by
//! [`crate::scheduler::cache::CacheManager`] and integration tests.

pub use crate::scheduler::cache::{
    CacheKey, CachedEntry, PrefixCache, PrefixCacheConfig, PrefixCacheStats, hash_tokens,
};
pub use crate::scheduler::memory::BlockAllocator;

pub use vllm_traits::BLOCK_SIZE;
