pub mod prefix_cache;

pub use prefix_cache::{CacheKey, CachedEntry, PrefixCache, PrefixCacheStats, hash_tokens};

/// Cache manager for prefix caching of KV blocks.
pub struct CacheManager {
    prefix_cache: PrefixCache,
}

impl CacheManager {
    /// Creates a new empty cache manager.
    pub fn new() -> Self {
        Self {
            prefix_cache: PrefixCache::new(),
        }
    }

    /// Looks up a cached entry by key.
    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry> {
        self.prefix_cache.get(key)
    }

    /// Inserts a new entry into the cache.
    pub fn insert(&mut self, key: CacheKey, blocks: Vec<usize>, token_count: usize) {
        self.prefix_cache.insert(key, blocks, token_count);
    }

    /// Inserts an entry with shared block references.
    pub fn insert_arc(
        &mut self,
        key: CacheKey,
        blocks: std::sync::Arc<Vec<usize>>,
        token_count: usize,
    ) {
        self.prefix_cache.insert_arc(key, blocks, token_count);
    }

    /// Finds a cached prefix that matches the given tokens.
    pub fn find_prefix_match(&mut self, tokens: &[u32]) -> Option<&CachedEntry> {
        self.prefix_cache.find_prefix_match(tokens)
    }

    /// Finds if any cached entry is a prefix of the given tokens.
    pub fn find_reverse_prefix_match(
        &self,
        tokens: &[u32],
    ) -> Option<(std::sync::Arc<Vec<usize>>, usize)> {
        self.prefix_cache.find_reverse_prefix_match(tokens)
    }

    /// Evicts the least recently used entry.
    pub fn evict(&mut self, allocator: &mut super::memory::BlockAllocator) {
        self.prefix_cache.evict(allocator);
    }

    /// Returns the number of entries in the cache.
    pub fn len(&self) -> usize {
        self.prefix_cache.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.prefix_cache.is_empty()
    }

    /// Returns true if the key exists in the cache.
    pub fn contains_key(&self, key: &CacheKey) -> bool {
        self.prefix_cache.contains_key(key)
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> PrefixCacheStats {
        self.prefix_cache.stats()
    }

    /// Returns the cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.prefix_cache.hit_rate()
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}
