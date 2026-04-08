pub mod prefix_cache;

pub use prefix_cache::{hash_tokens, CacheKey, CachedEntry, PrefixCache, PrefixCacheStats};

pub struct CacheManager {
    prefix_cache: PrefixCache,
}

impl CacheManager {
    pub fn new() -> Self {
        Self {
            prefix_cache: PrefixCache::new(),
        }
    }

    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry> {
        self.prefix_cache.get(key)
    }

    pub fn insert(&mut self, key: CacheKey, blocks: Vec<usize>, token_count: usize) {
        self.prefix_cache.insert(key, blocks, token_count);
    }

    pub fn insert_arc(
        &mut self,
        key: CacheKey,
        blocks: std::sync::Arc<Vec<usize>>,
        token_count: usize,
    ) {
        self.prefix_cache.insert_arc(key, blocks, token_count);
    }

    pub fn find_prefix_match(&mut self, tokens: &[u32]) -> Option<&CachedEntry> {
        self.prefix_cache.find_prefix_match(tokens)
    }

    pub fn find_reverse_prefix_match(
        &self,
        tokens: &[u32],
    ) -> Option<(std::sync::Arc<Vec<usize>>, usize)> {
        self.prefix_cache.find_reverse_prefix_match(tokens)
    }

    pub fn evict(&mut self, allocator: &mut super::memory::BlockAllocator) {
        self.prefix_cache.evict(allocator);
    }

    pub fn len(&self) -> usize {
        self.prefix_cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prefix_cache.is_empty()
    }

    pub fn contains_key(&self, key: &CacheKey) -> bool {
        self.prefix_cache.contains_key(key)
    }

    pub fn stats(&self) -> PrefixCacheStats {
        self.prefix_cache.stats()
    }

    pub fn hit_rate(&self) -> f64 {
        self.prefix_cache.hit_rate()
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}
