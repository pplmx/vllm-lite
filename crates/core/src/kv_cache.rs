use crate::types::{BlockId, TokenId};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub const BLOCK_SIZE: usize = 16;

pub type CacheKey = u64;

pub fn hash_tokens(tokens: &[TokenId]) -> CacheKey {
    tokens
        .iter()
        .fold(0u64, |acc, &t| acc.wrapping_mul(31).wrapping_add(t as u64))
}

#[derive(Clone)]
pub struct CachedEntry {
    pub key: CacheKey,
    pub blocks: Vec<BlockId>,
    pub token_count: usize,
    pub last_access: Instant,
}

pub struct PrefixCache {
    entries: HashMap<CacheKey, CachedEntry>,
    lru_order: VecDeque<CacheKey>,
    block_refs: HashMap<BlockId, usize>,
    prefix_match_cache: HashMap<CacheKey, CacheKey>,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            block_refs: HashMap::new(),
            prefix_match_cache: HashMap::new(),
        }
    }

    pub fn get(&mut self, key: CacheKey) -> Option<&CachedEntry> {
        if let Some(entry) = self.entries.get(&key) {
            self.lru_order.retain(|k| *k != key);
            self.lru_order.push_front(key);
            Some(entry)
        } else {
            None
        }
    }

    pub fn insert(&mut self, key: CacheKey, blocks: Vec<BlockId>, token_count: usize) {
        if let Some(old_entry) = self.entries.remove(&key) {
            for &block in &old_entry.blocks {
                if let Some(count) = self.block_refs.get_mut(&block) {
                    *count -= 1;
                    if *count == 0 {
                        self.block_refs.remove(&block);
                    }
                }
            }
        }

        for &block in &blocks {
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
    }

    pub fn evict(&mut self, allocator: &mut BlockAllocator) {
        while let Some(oldest_key) = self.lru_order.pop_back() {
            if let Some(entry) = self.entries.remove(&oldest_key) {
                for &block in &entry.blocks {
                    if let Some(count) = self.block_refs.get_mut(&block) {
                        *count -= 1;
                        if *count == 0 {
                            allocator.free(&[block]);
                            self.block_refs.remove(&block);
                        }
                    }
                }
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
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct BlockAllocator {
    num_blocks: usize,
    free_list: Vec<BlockId>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize) -> Self {
        let free_list: Vec<BlockId> = (0..num_blocks).collect();
        Self {
            num_blocks,
            free_list,
        }
    }

    pub fn allocate(&mut self, num_blocks: usize) -> Option<Vec<BlockId>> {
        if self.free_list.len() >= num_blocks {
            Some(
                (0..num_blocks)
                    .map(|_| self.free_list.pop().unwrap())
                    .collect(),
            )
        } else {
            None
        }
    }

    pub fn free(&mut self, blocks: &[BlockId]) {
        for &block in blocks {
            self.free_list.push(block);
        }
    }

    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    pub fn total(&self) -> usize {
        self.num_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let mut alloc = BlockAllocator::new(10);

        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(alloc.available(), 7);

        alloc.free(&blocks);
        assert_eq!(alloc.available(), 10);
    }

    #[test]
    fn test_oom() {
        let mut alloc = BlockAllocator::new(2);
        alloc.allocate(2).unwrap();
        assert!(alloc.allocate(1).is_none());
    }

    #[test]
    fn test_free_order() {
        let mut alloc = BlockAllocator::new(5);

        let blocks1 = alloc.allocate(2).unwrap();
        let blocks2 = alloc.allocate(2).unwrap();

        alloc.free(&blocks2);
        alloc.free(&blocks1);

        // Should have all 5 back (4 allocated + 1 never used)
        assert_eq!(alloc.available(), 5);
    }

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
    fn test_block_allocation_exact_fit() {
        let mut alloc = BlockAllocator::new(3);
        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(alloc.available(), 0);
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
}
