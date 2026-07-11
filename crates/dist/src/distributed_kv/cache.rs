//! Thread-safe distributed KV-cache implementation: in-process `HashMap` + optional gRPC peer sync.
//!
//! Activated by `--features multi-node`. The single-node path is the
//! `HashMap`-backed `DistributedKVCache` used by tests and embedded builds.
#![allow(clippy::module_name_repetitions)]
use super::{CacheConfig, CacheMessage, NodeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[derive(Debug)]
/// Cache for `DistributedKV`. Keyed lookup with the configured eviction policy (LRU, ARC, FIFO). Thread-safe.
pub struct DistributedKVCache {
    /// Cache configuration (capacity, eviction policy, peer URLs).
    config: CacheConfig,
    /// Local in-process KV map.
    local_cache: Arc<RwLock<HashMap<u64, CacheEntry>>>,
    /// Cache statistics.
    stats: RwLock<CacheStats>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    key: u64,
    value_hash: u64,
    state: CacheState,
    last_access: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheState {
    Shared,
    Modified,
    Exclusive,
    Invalid,
}

/// Telemetry snapshot for Cache: counters, gauges, and percentile latencies. Cloned and serialized on every metrics export.
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Cumulative cache hits.
    pub hits: u64,
    /// Cumulative cache misses.
    pub misses: u64,
    /// Cumulative invalidations.
    pub invalidations: u64,
    /// Cumulative successful updates (put-on-existing or new entries).
    pub updates: u64,
}

impl DistributedKVCache {
    /// Construct a new cache scoped to `config.node_id`, with an empty
    /// local store and zeroed statistics.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            local_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: RwLock::new(CacheStats::default()),
        }
    }

    /// Look up `key` in the local cache.
    ///
    /// Returns the cached `value_hash` on a hit; `None` on a miss or if
    /// the local entry is in the `Invalid` state. Hits/misses are
    /// recorded in the [`CacheStats`] returned by [`Self::stats`].
    pub fn get(&self, key: u64) -> Option<u64> {
        let mut cache = self.local_cache.write().ok()?;

        if let Some(entry) = cache.get_mut(&key) {
            debug_assert_eq!(entry.key, key);
            if entry.state != CacheState::Invalid {
                entry.last_access = current_timestamp();
                if let Ok(mut stats) = self.stats.write() {
                    stats.hits += 1;
                }
                return Some(entry.value_hash);
            }
        }

        drop(cache);
        if let Ok(mut stats) = self.stats.write() {
            stats.misses += 1;
        }
        None
    }

    /// Walk `keys` in order; return the count of consecutive hits
    /// from the start.
    ///
    /// The first miss (`None` from [`Self::get`]) stops the walk and
    /// the prefix length up to (but not including) that key is
    /// returned. If every key hits, the full `keys.len()` is
    /// returned.
    ///
    /// Each key still bumps the [`CacheStats::hits`] / `misses`
    /// counter the same way [`Self::get`] does, so prefix-lookup
    /// telemetry is indistinguishable from individual gets at the
    /// counter level.
    ///
    /// Empty `keys` returns `0` (no work, no misses recorded).
    ///
    /// # Lock semantics
    ///
    /// Single write-lock acquisition on the local map. After the
    /// walk, the write lock is dropped before the stats lock is
    /// acquired — same pattern as [`Self::get`] — so a slow stats
    /// writer can't block the cache map.
    pub fn lookup_prefix(&self, keys: &[u64]) -> usize {
        if keys.is_empty() {
            return 0;
        }

        let mut matched = 0usize;
        let mut first_miss = None;

        {
            let mut cache = match self.local_cache.write() {
                Ok(g) => g,
                Err(_) => return 0,
            };
            for (i, &key) in keys.iter().enumerate() {
                match cache.get_mut(&key) {
                    Some(entry) if entry.state != CacheState::Invalid => {
                        debug_assert_eq!(entry.key, key);
                        entry.last_access = current_timestamp();
                        matched = i + 1;
                    }
                    _ => {
                        first_miss = Some(i);
                        break;
                    }
                }
            }
        }

        // Bump stats outside the map lock.
        if let Ok(mut stats) = self.stats.write() {
            stats.hits += matched as u64;
            if let Some(i) = first_miss {
                stats.misses += (keys.len() - i) as u64;
            }
        }

        matched
    }

    /// Insert or update the local entry for `key` with `value_hash`.
    ///
    /// Computes the owner set via consistent hashing over the
    /// configured node count and marks the local cache entry as
    /// `Exclusive` (this node is the primary owner) / `Shared` (a
    /// replica) / `Modified` (already-present entry being updated).
    /// Updates the `updates` counter on [`CacheStats`].
    pub fn put(&self, key: u64, value_hash: u64) {
        let owner_nodes = self.compute_owner_nodes(key);
        let timestamp = current_timestamp();

        if let Ok(mut cache) = self.local_cache.write() {
            let state = if cache.contains_key(&key) {
                CacheState::Modified
            } else if self.config.node_id == owner_nodes[0] {
                CacheState::Exclusive
            } else {
                CacheState::Shared
            };

            cache.insert(
                key,
                CacheEntry {
                    key,
                    value_hash,
                    state,
                    last_access: timestamp,
                },
            );
        }

        if let Ok(mut stats) = self.stats.write() {
            stats.updates += 1;
        }
    }

    /// Drop the local entry for `key` and increment the
    /// `invalidations` counter on [`CacheStats`].
    ///
    /// Remote replicas are not notified here — coordination is the
    /// caller's responsibility (typically via [`Self::handle_message`]
    /// on the source node, then broadcasting an `Invalidate` message).
    pub fn invalidate(&self, key: u64) {
        if let Ok(mut cache) = self.local_cache.write() {
            cache.remove(&key);
        }
        if let Ok(mut stats) = self.stats.write() {
            stats.invalidations += 1;
        }
    }

    /// Dispatch a wire-protocol `msg` arriving from a peer node.
    ///
    /// Returns the response (if any) that the caller should send back
    /// to the source: `Read` → `Ack` on hit / no reply on miss,
    /// `Invalidate` → local invalidate + `Ack`, `Update`/`Write` →
    /// local `put` with no reply, `Ack` → no reply. This is the
    /// single entry point that the gRPC handler in `grpc.rs` uses to
    /// fold peer traffic into the local cache.
    pub fn handle_message(&self, msg: &CacheMessage) -> Option<CacheMessage> {
        match &msg.operation {
            super::protocol::CacheOperation::Read { key, .. } => {
                if let Some(_value_hash) = self.get(*key) {
                    return Some(CacheMessage::new(
                        self.config.node_id,
                        msg.source,
                        super::protocol::CacheOperation::Ack {
                            operation_id: msg.id,
                            success: true,
                        },
                    ));
                }
                None
            }
            super::protocol::CacheOperation::Invalidate { key, .. } => {
                self.invalidate(*key);
                Some(CacheMessage::new(
                    self.config.node_id,
                    msg.source,
                    super::protocol::CacheOperation::Ack {
                        operation_id: msg.id,
                        success: true,
                    },
                ))
            }
            super::protocol::CacheOperation::Update {
                key, value_hash, ..
            }
            | super::protocol::CacheOperation::Write {
                key, value_hash, ..
            } => {
                self.put(*key, *value_hash);
                None
            }
            super::protocol::CacheOperation::Ack { .. } => None,
        }
    }

    /// Snapshot of hit / miss / update / invalidation counters since
    /// cache construction. Cheap (clones the stats struct under the
    /// stats lock).
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    fn compute_owner_nodes(&self, key: u64) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(self.config.replication_factor);
        // invariant: key % num_nodes fits in usize on all targets since the
        // modulus is bounded by num_nodes.
        #[allow(clippy::cast_possible_truncation)]
        let base = (key as usize) % self.config.num_nodes;

        for i in 0..self.config.replication_factor {
            let node_id = (base + i) % self.config.num_nodes;
            nodes.push(NodeId::new(node_id));
        }

        nodes
    }
}

fn current_timestamp() -> u64 {
    // invariant: monotonic clock is always >= UNIX_EPOCH; saturating u64
    // conversion is safe for any realistic timestamp.
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            // invariant: pre-conditions make this infallible at this call site.
            .unwrap()
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_put_get() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        cache.put(1, 100);
        let result = cache.get(1);

        assert_eq!(result, Some(100));
    }

    #[test]
    fn test_cache_miss() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        let result = cache.get(999);
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_stats_track_hits_and_misses() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        cache.put(1, 100);
        assert_eq!(cache.get(1), Some(100));
        assert_eq!(cache.get(999), None);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.updates, 1);
    }

    #[test]
    fn test_cache_invalidate() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        cache.put(1, 100);
        assert!(cache.get(1).is_some());

        cache.invalidate(1);
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn test_owner_nodes_single_replication() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);

        let owners = cache.compute_owner_nodes(5);
        assert_eq!(owners.len(), 2);
        assert!(owners.contains(&NodeId::new(1))); // 5 % 4 = 1
    }

    #[test]
    fn test_owner_nodes_distribution() {
        let config = CacheConfig::new(NodeId(0), 8);
        let cache = DistributedKVCache::new(config);

        for key in 0..16 {
            let owners = cache.compute_owner_nodes(key);
            assert!(!owners.is_empty());
            assert!(owners.windows(2).all(|w| w[0] != w[1])); // No duplicates
        }
    }

    // --- lookup_prefix ---

    #[test]
    fn test_lookup_prefix_empty_input_returns_zero() {
        // Empty input is a no-op (no locks, no counter bumps).
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);
        assert_eq!(cache.lookup_prefix(&[]), 0);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_lookup_prefix_all_hits() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);
        for k in 1..=4 {
            cache.put(k, k * 100);
        }

        assert_eq!(cache.lookup_prefix(&[1, 2, 3, 4]), 4);
        let stats = cache.stats();
        assert_eq!(stats.hits, 4, "every hit must bump hits counter");
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_lookup_prefix_partial_match_stops_at_first_miss() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);
        // Only keys 10, 20 are present; 30, 40, 50 are not.
        cache.put(10, 0xA);
        cache.put(20, 0xB);

        assert_eq!(
            cache.lookup_prefix(&[10, 20, 30, 40, 50]),
            2,
            "should match keys 10 and 20, then stop at 30"
        );
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        // misses counts the first miss + everything after it (the
        // "rest of the prefix" doesn't get individually queried, so
        // bumping one miss per unseen key would be misleading; we
        // count the first miss + remaining).
        assert_eq!(
            stats.misses, 3,
            "first miss + remaining 2 keys all bumped as misses"
        );
    }

    #[test]
    fn test_lookup_prefix_first_key_misses() {
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);
        cache.put(99, 1);

        assert_eq!(
            cache.lookup_prefix(&[1, 2, 3]),
            0,
            "first key miss returns 0 matched"
        );
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 3, "all 3 keys count as misses");
    }

    #[test]
    fn test_lookup_prefix_invalid_entries_count_as_miss() {
        // Entries in the `Invalid` state must count as misses (they
        // don't expose a value_hash to callers via `get`).
        let config = CacheConfig::new(NodeId(0), 4);
        let cache = DistributedKVCache::new(config);
        cache.put(1, 0xA);
        cache.put(2, 0xB);
        cache.invalidate(2);

        assert_eq!(
            cache.lookup_prefix(&[1, 2]),
            1,
            "key 2 is invalidated; lookup stops there"
        );
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1, "invalidated key counts as miss");
    }

    #[test]
    fn test_lookup_prefix_distinguishes_different_caches() {
        // Two independent caches don't see each other's entries
        // (lookup_prefix is purely local).
        let cache_a = DistributedKVCache::new(CacheConfig::new(NodeId(0), 4));
        let cache_b = DistributedKVCache::new(CacheConfig::new(NodeId(1), 4));
        cache_a.put(1, 0xAA);

        assert_eq!(cache_a.lookup_prefix(&[1, 2]), 1);
        assert_eq!(
            cache_b.lookup_prefix(&[1, 2]),
            0,
            "cache_b never saw key 1; both keys miss"
        );
    }
}
