use crate::types::{BlockId, TokenId};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use super::BlockAllocator;

pub type CacheKey = u64;

pub fn hash_tokens(tokens: &[TokenId]) -> CacheKey {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone)]
pub struct CachedEntry {
    pub key: CacheKey,
    pub blocks: Arc<Vec<BlockId>>,
    pub token_count: usize,
    pub last_access: Instant,
}

#[derive(Clone, Default)]
pub struct PrefixCacheStats {
    pub entries: usize,
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
}

pub struct PrefixCache {
    entries: HashMap<CacheKey, CachedEntry>,
    lru_order: VecDeque<CacheKey>,
    block_refs: HashMap<BlockId, usize>,
    prefix_match_cache: HashMap<CacheKey, CacheKey>,
    stats: PrefixCacheStats,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            block_refs: HashMap::new(),
            prefix_match_cache: HashMap::new(),
            stats: PrefixCacheStats::default(),
        }
    }

    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry> {
        if let Some(entry) = self.entries.get(&key) {
            self.lru_order.retain(|k| *k != key);
            self.lru_order.push_front(key);
            self.stats.hits += 1;
            Some(entry)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn insert(&mut self, key: CacheKey, blocks: Vec<BlockId>, token_count: usize) {
        self.insert_arc(key, Arc::new(blocks), token_count);
    }

    pub fn insert_arc(&mut self, key: CacheKey, blocks: Arc<Vec<BlockId>>, token_count: usize) {
        if let Some(old_entry) = self.entries.remove(&key) {
            for &block in old_entry.blocks.as_ref() {
                if let Some(count) = self.block_refs.get_mut(&block) {
                    *count -= 1;
                    if *count == 0 {
                        self.block_refs.remove(&block);
                    }
                }
            }
        }

        for &block in blocks.as_ref() {
            *self.block_refs.entry(block).or_insert(0) += 1;
        }

        let entry = CachedEntry {
            key,
            blocks,
            token_count,
            last_access: Instant::now(),
        };
        self.entries.insert(key, entry);
        self.lru_order.retain(|k| *k != key);
        self.lru_order.push_front(key);
        self.prefix_match_cache.clear();
        self.stats.entries = self.entries.len();
    }

    pub fn evict(&mut self, allocator: &mut BlockAllocator) {
        while let Some(oldest_key) = self.lru_order.pop_back() {
            if let Some(entry) = self.entries.remove(&oldest_key) {
                for &block in entry.blocks.as_ref() {
                    if let Some(count) = self.block_refs.get_mut(&block) {
                        *count -= 1;
                        if *count == 0 {
                            allocator.free(&[block]);
                            self.block_refs.remove(&block);
                        }
                    }
                }
                self.stats.evictions += 1;
                self.stats.entries = self.entries.len();
                break;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains_key(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    pub fn find_prefix_match(&mut self, tokens: &[TokenId]) -> Option<&CachedEntry> {
        if tokens.is_empty() {
            return None;
        }

        let query_key = hash_tokens(tokens);
        if let Some(&matched_key) = self.prefix_match_cache.get(&query_key) {
            self.lru_order.retain(|k| *k != matched_key);
            self.lru_order.push_front(matched_key);
            return self.entries.get(&matched_key);
        }

        for prefix_len in (1..=tokens.len()).rev() {
            let prefix = &tokens[..prefix_len];
            let key = hash_tokens(prefix);
            if let Some(entry) = self.entries.get(&key) {
                self.prefix_match_cache.insert(query_key, key);
                return Some(entry);
            }
        }
        None
    }

    pub fn stats(&self) -> PrefixCacheStats {
        self.stats.clone()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tokens() {
        assert_eq!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 3]));
        assert_ne!(hash_tokens(&[1, 2, 3]), hash_tokens(&[1, 2, 4]));
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = PrefixCache::new();
        cache.insert(123, vec![1, 2], 2);

        assert!(cache.get(123).is_some());
        assert!(cache.get(456).is_none());
    }

    #[test]
    fn test_lru_order() {
        let mut cache = PrefixCache::new();
        cache.insert(1, vec![1], 1);
        cache.insert(2, vec![2], 1);
        cache.insert(3, vec![3], 1);

        cache.get(1);

        let mut alloc = BlockAllocator::new(10);
        cache.evict(&mut alloc);

        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_shared_block_reference_count() {
        let mut cache = PrefixCache::new();

        cache.insert(1, vec![1, 2, 3], 3);
        cache.insert(2, vec![1, 2, 3], 3);

        let mut alloc = BlockAllocator::new(10);
        cache.evict(&mut alloc);

        assert!(cache.get(2).is_some());
    }

    #[test]
    fn test_multiple_evictions_release_all_blocks() {
        let mut cache = PrefixCache::new();

        cache.insert(1, vec![1, 2], 2);
        cache.insert(2, vec![3, 4], 2);
        assert_eq!(cache.len(), 2);

        cache.evict(&mut BlockAllocator::new(10));
        assert_eq!(cache.len(), 1);

        cache.evict(&mut BlockAllocator::new(10));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_get_updates_lru() {
        let mut cache = PrefixCache::new();
        cache.insert(1, vec![1], 1);
        cache.insert(2, vec![2], 1);

        cache.get(1);
        cache.get(2);
        cache.get(1);

        let mut alloc = BlockAllocator::new(10);
        cache.evict(&mut alloc);

        assert!(cache.get(2).is_none());
        assert!(cache.get(1).is_some());
    }

    #[test]
    fn test_cache_contains_key() {
        let mut cache = PrefixCache::new();
        cache.insert(100, vec![1], 1);

        assert!(cache.contains_key(&100));
        assert!(!cache.contains_key(&200));
    }

    #[test]
    fn test_cache_is_empty() {
        let cache = PrefixCache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_prefix_match() {
        let mut cache = PrefixCache::new();

        cache.insert(hash_tokens(&[1, 2]), vec![1], 2);

        let result = cache.find_prefix_match(&[1, 2, 3]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().token_count, 2);

        let result = cache.find_prefix_match(&[3, 4]);
        assert!(result.is_none());
    }

    #[test]
    fn test_stats() {
        let mut cache = PrefixCache::new();

        cache.insert(1, vec![1], 1);
        assert_eq!(cache.stats().entries, 1);

        cache.get(1);
        cache.get(2);

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.hit_rate() - 0.5).abs() < 0.01);
    }
}
