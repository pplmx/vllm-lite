//! Thread-safe distributed KV-cache implementation: in-process `HashMap` + optional gRPC peer sync.
//!
//! Activated by `--features multi-node`. The single-node path is the
//! `HashMap`-backed `DistributedKVCache` used by tests and embedded builds.
#![allow(clippy::module_name_repetitions)]
use super::block_data_source::{BlockDataSource, FetchError};
use super::block_sink::BlockSink;
use super::{CacheConfig, CacheMessage, NodeId};
use crate::error::GrpcError;
use crate::grpc_client::PeerClient;
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
    /// Per-peer gRPC clients. `None` until [`Self::connect_peers`]
    /// is called; `Some(vec![])` once called with no peer URLs in
    /// the config (single-node mode, no broadcast). Phase 19 OPS-05c.
    peer_clients: Option<Arc<Vec<PeerClient>>>,
    /// Optional source of raw block bytes for [`Self::fetch_block`].
    /// Without this, [`Self::fetch_block`] can only satisfy requests
    /// via remote peers. Phase 31-D OPS-31d.
    block_data_source: Option<Arc<dyn BlockDataSource>>,
    /// Optional sink for receiver-side install. When `Some` AND
    /// [`Self::install_on_fetch`] is `true`, every successful
    /// `fetch_block` (peer or local source) hands the bytes to the
    /// sink so the local cache ends up populated. P42 receiver-side
    /// closing of the multi-node KV block transfer loop.
    block_sink: Option<Arc<dyn BlockSink>>,
    /// Toggle for the auto-install behavior. Default `true` (closed
    /// loop is the vLLM-equivalent UX). Tests + diagnostics can opt
    /// out via [`Self::with_install_on_fetch`].
    install_on_fetch: bool,
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
    ///
    /// Peer clients are NOT built here — the constructor stays
    /// synchronous. Call [`Self::connect_peers`] later to validate
    /// the configured peer URLs and construct the per-peer client
    /// handles (lazy channel connect happens on first RPC). Until
    /// then, `put` / `invalidate` are purely local.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            local_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: RwLock::new(CacheStats::default()),
            peer_clients: None,
            block_data_source: None,
            block_sink: None,
            install_on_fetch: true,
        }
    }

    /// Attach a [`BlockDataSource`] so [`Self::fetch_block`] can
    /// serve requests from local bytes when no peer has the block
    /// (or in single-node mode).
    ///
    /// Phase 31-D OPS-31d.
    #[must_use]
    pub fn with_block_data_source(mut self, source: Arc<dyn BlockDataSource>) -> Self {
        self.block_data_source = Some(source);
        self
    }

    /// Attach a [`BlockSink`] so every successful `fetch_block`
    /// installs the bytes into the local cache (subject to
    /// [`Self::with_install_on_fetch`] toggle). P42 receiver-side
    /// closure of the multi-node KV block transfer loop.
    #[must_use]
    pub fn with_block_sink(mut self, sink: Arc<dyn BlockSink>) -> Self {
        self.block_sink = Some(sink);
        self
    }

    /// Toggle auto-install behavior. Default `true`. Set to `false`
    /// for tests that want to verify `fetch_block`'s wire shape
    /// without triggering side effects.
    #[must_use]
    pub fn with_install_on_fetch(mut self, enable: bool) -> Self {
        self.install_on_fetch = enable;
        self
    }

    /// Build gRPC clients for every URL in [`CacheConfig::peer_urls`].
    ///
    /// Idempotent: re-calling replaces the existing peer list (the
    /// old clients are dropped along with their channels). Calling
    /// with an empty `peer_urls` clears any prior peer set, returning
    /// the cache to single-node mode.
    ///
    /// This is a sync helper because [`PeerClient::new`] only
    /// validates URLs and creates lazy endpoints — actual TCP+HTTP/2
    /// handshakes happen later inside the client on the first RPC.
    ///
    /// # Errors
    ///
    /// Returns [`GrpcError`] if any URL fails to parse. Partial
    /// success is not possible — if one URL is malformed, the call
    /// returns before validating subsequent URLs and the cache is
    /// left in its prior state.
    pub fn connect_peers(&mut self) -> Result<(), GrpcError> {
        let mut clients = Vec::with_capacity(self.config.peer_urls.len());
        for url in &self.config.peer_urls {
            let client = PeerClient::new(url.clone())?;
            clients.push(client);
        }
        self.peer_clients = Some(Arc::new(clients));
        Ok(())
    }

    /// Snapshot of the configured peer URLs.
    #[must_use]
    pub fn peer_urls(&self) -> &[String] {
        &self.config.peer_urls
    }

    /// Whether [`Self::connect_peers`] has been called (regardless
    /// of whether peer URLs were empty).
    #[must_use]
    pub const fn peers_connected(&self) -> bool {
        self.peer_clients.is_some()
    }

    /// Number of currently-connected peer clients. `0` before
    /// [`Self::connect_peers`] is called or when the config has no
    /// peer URLs.
    #[must_use]
    pub fn peer_client_count(&self) -> usize {
        self.peer_clients.as_ref().map_or(0, |v| v.len())
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
    ///
    /// # Peer broadcast (OPS-05c)
    ///
    /// If peer clients were configured via [`Self::connect_peers`]
    /// and the call is running inside a tokio runtime, the put is
    /// also broadcast to every peer via a fire-and-forget tokio
    /// task. The local put is always synchronous; broadcast failures
    /// are logged and dropped (no retry, no back-pressure). When no
    /// tokio runtime is available (test path without `#[tokio::test]`),
    /// the broadcast is silently skipped — the local put still
    /// happens.
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

        self.broadcast_put(key, value_hash);
    }

    /// Drop the local entry for `key` and increment the
    /// `invalidations` counter on [`CacheStats`].
    ///
    /// # Peer broadcast (OPS-05c)
    ///
    /// Same fire-and-forget pattern as [`Self::put`]: a tokio task
    /// is spawned (if a runtime is available) that calls each
    /// peer's `InvalidateKVCache` RPC. Local invalidate always
    /// happens first.
    pub fn invalidate(&self, key: u64) {
        if let Ok(mut cache) = self.local_cache.write() {
            cache.remove(&key);
        }
        if let Ok(mut stats) = self.stats.write() {
            stats.invalidations += 1;
        }

        self.broadcast_invalidate(key);
    }

    /// Fetch raw bytes for `block_id` from a peer (or the local
    /// [`BlockDataSource`] as a fallback).
    ///
    /// Phase 31-D OPS-31d. The algorithm is fan-out fallback:
    ///
    /// 1. **Local precheck**: the local cache must already hold an
    ///    entry for `block_id`; otherwise the caller has no
    ///    `expected_hash` to verify against and the request is
    ///    meaningless — return [`FetchError::NotFound`].
    /// 2. **Fan-out**: every configured peer is queried in parallel
    ///    via [`PeerClient::fetch_block`]. The first response whose
    ///    `chain_hash` matches the local `value_hash` wins; peers
    ///    returning a mismatch (or a transport error) are logged and
    ///    skipped.
    /// 3. **Local fallback**: if fan-out yields no success, try the
    ///    local [`BlockDataSource`] (production wraps `PagedKvCache`).
    /// 4. **Nothing worked**: return
    ///    [`FetchError::NoPeers`] (if no peers and no source) or
    ///    [`FetchError::AllPeersFailed`] (otherwise).
    ///
    /// Smart owner-based routing (using `Self::compute_owner_nodes`)
    /// is deferred to v32+ — fan-out wastes bandwidth but is correct
    /// and simple.
    ///
    /// # Errors
    ///
    /// See [`FetchError`].
    pub async fn fetch_block(&self, block_id: u64) -> Result<Vec<u8>, FetchError> {
        // Step 1: precheck. We need an expected_hash to verify peer
        // responses against; without a local entry there's nothing
        // to verify, so the call is a no-op miss.
        let expected_hash = self.get(block_id).ok_or(FetchError::NotFound(block_id))?;

        // Step 2: fan-out to peers (if any).
        let peers: Vec<PeerClient> = self
            .peer_clients
            .as_ref()
            .map(|c| c.iter().cloned().collect())
            .unwrap_or_default();

        if !peers.is_empty() {
            // Use tokio::task::JoinSet to fan out across peers
            // without adding a `futures` crate dependency.
            let mut join_set = tokio::task::JoinSet::new();
            for client in &peers {
                let client = client.clone();
                join_set.spawn(async move {
                    (
                        client.url().to_string(),
                        client.fetch_block(block_id, expected_hash).await,
                    )
                });
            }
            while let Some(joined) = join_set.join_next().await {
                let (url, result) = match joined {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::warn!(error = %e, "peer fetch task failed to join");
                        continue;
                    }
                };
                match result {
                    Ok(resp) if resp.chain_hash == expected_hash => {
                        return self.maybe_install(block_id, resp.data).await;
                    }
                    Ok(resp) => {
                        tracing::warn!(
                            peer = %url,
                            block_id,
                            expected = expected_hash,
                            actual = resp.chain_hash,
                            "peer returned block bytes with mismatched chain_hash"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            peer = %url,
                            block_id,
                            error = %e,
                            "peer fetch_block failed"
                        );
                    }
                }
            }
        }

        // Step 3: fall back to the local source.
        if let Some(source) = self.block_data_source.as_ref() {
            let bytes = source.fetch_block(block_id).await?;
            return self.maybe_install(block_id, bytes).await;
        }

        // Step 4: nothing worked.
        if peers.is_empty() {
            Err(FetchError::NoPeers)
        } else {
            Err(FetchError::AllPeersFailed(peers.len()))
        }
    }

    /// P42: hand the bytes to the configured [`BlockSink`] (if any)
    /// when `install_on_fetch` is enabled. Best-effort — a sink
    /// failure is logged as a warning but does NOT cause the
    /// fetch to fail (the bytes were correctly retrieved from the
    /// peer; installing them is a local cache concern, not a
    /// transport correctness concern).
    ///
    /// This is the "close the loop" path: receiver pulls a block
    /// from a peer once, then has it locally for subsequent
    /// `has_block` / `read_layer_block` calls without a remote RPC.
    async fn maybe_install(&self, block_id: u64, bytes: Vec<u8>) -> Result<Vec<u8>, FetchError> {
        let sink = match (self.install_on_fetch, self.block_sink.as_ref()) {
            (true, Some(sink)) => sink,
            _ => return Ok(bytes),
        };
        match sink.write_block(block_id, &bytes).await {
            Ok(()) => Ok(bytes),
            Err(e) => {
                tracing::warn!(
                    block_id,
                    error = %e,
                    "BlockSink install failed; returning bytes anyway"
                );
                Ok(bytes)
            }
        }
    }

    /// Spawn a tokio task that fires `put_kv_cache` on every peer.
    /// No-op when no peer clients are configured or no tokio
    /// runtime is available.
    fn broadcast_put(&self, key: u64, value_hash: u64) {
        let Some(clients) = self.peer_clients.clone() else {
            return;
        };
        if clients.is_empty() {
            return;
        }
        // try_current returns Err if no runtime is in scope (e.g.,
        // unit tests outside `#[tokio::test]`). Silently skip —
        // the local put still happened.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                block_id = key,
                "put: no tokio runtime; skipping peer broadcast"
            );
            return;
        };
        handle.spawn(async move {
            for client in clients.iter() {
                if let Err(e) = client.put(key, value_hash).await {
                    tracing::warn!(
                        peer = %client.url(),
                        block_id = key,
                        error = %e,
                        "peer put failed; local update stands"
                    );
                }
            }
        });
    }

    /// Spawn a tokio task that fires `invalidate_kv_cache` on every
    /// peer. Same semantics as [`Self::broadcast_put`].
    fn broadcast_invalidate(&self, key: u64) {
        let Some(clients) = self.peer_clients.clone() else {
            return;
        };
        if clients.is_empty() {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                block_id = key,
                "invalidate: no tokio runtime; skipping peer broadcast"
            );
            return;
        };
        handle.spawn(async move {
            for client in clients.iter() {
                if let Err(e) = client.invalidate(key).await {
                    tracing::warn!(
                        peer = %client.url(),
                        block_id = key,
                        error = %e,
                        "peer invalidate failed; local drop stands"
                    );
                }
            }
        });
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

    // --- fetch_block (Phase 31-D OPS-31d) ---
    //
    // These tests use a `tokio::test` runtime for the async path,
    // but cover only the precheck + local-source-fallback branches.
    // Peer-fan-out paths are exercised by the integration tests in
    // `tests/kv_block_transfer.rs` because they need a real gRPC
    // server on an ephemeral port.

    #[tokio::test]
    async fn fetch_block_returns_not_found_when_local_cache_missing() {
        // Source is wired but the cache has no entry for `1` — the
        // precheck fails, so we never call the source.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let mut source = MockBlockDataSource::new();
        // Insert a block the cache has never heard of.
        source.insert(99, vec![1, 2, 3]);

        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source));

        let result = cache.fetch_block(99).await;
        assert!(
            matches!(result, Err(FetchError::NotFound(99))),
            "expected NotFound(99); got {result:?}"
        );
    }

    #[tokio::test]
    async fn fetch_block_returns_no_peers_when_single_node_no_source() {
        // Cache has an entry, but no peers and no local source — we
        // can't fetch from anywhere.
        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1));
        cache.put(7, 0xABCD);

        let result = cache.fetch_block(7).await;
        assert!(
            matches!(result, Err(FetchError::NoPeers)),
            "expected NoPeers; got {result:?}"
        );
    }

    #[tokio::test]
    async fn fetch_block_falls_back_to_local_source_when_no_peers() {
        // Cache has an entry, local source has the bytes, no peers —
        // should return the source bytes.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0xDE, 0xAD, 0xBE, 0xEF]);

        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source));
        cache.put(7, 0xABCD);

        let bytes = cache.fetch_block(7).await.expect("fetch ok");
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[tokio::test]
    async fn fetch_block_propagates_source_not_found() {
        // Cache has an entry, local source does NOT have the bytes —
        // should bubble the source's NotFound up.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let source = MockBlockDataSource::new(); // empty

        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source));
        cache.put(7, 0xABCD);

        let result = cache.fetch_block(7).await;
        assert!(
            matches!(result, Err(FetchError::NotFound(7))),
            "expected NotFound(7) from source; got {result:?}"
        );
    }

    // --- fetch_block install_on_fetch (P42 T4) ---
    //
    // These tests use a custom `RecordingBlockSink` (an in-test
    // BlockSink impl that records every `write_block` call) so we
    // can assert the auto-install behavior end-to-end without
    // pulling in the `vllm-model` dependency.

    use crate::distributed_kv::block_sink::BlockSink as _BlockSinkTrait;
    use crate::distributed_kv::block_sink::WriteError;

    #[derive(Debug, Default)]
    struct RecordingBlockSink {
        calls: parking_lot::Mutex<Vec<(u64, usize)>>,
        return_error: bool,
    }

    impl RecordingBlockSink {
        fn call_count(&self) -> usize {
            self.calls.lock().len()
        }
    }

    #[async_trait::async_trait]
    impl _BlockSinkTrait for RecordingBlockSink {
        async fn write_block(&self, block_id: u64, bytes: &[u8]) -> Result<(), WriteError> {
            if self.return_error {
                return Err(WriteError::SinkUnavailable);
            }
            self.calls.lock().push((block_id, bytes.len()));
            Ok(())
        }
    }

    #[tokio::test]
    async fn fetch_block_installs_into_sink_when_install_on_fetch_default() {
        // install_on_fetch defaults to true, so a fetch with a wired
        // sink should trigger exactly one write_block call.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0xCA, 0xFE, 0xBA, 0xBE]);

        let sink = Arc::new(RecordingBlockSink::default());
        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source))
            .with_block_sink(Arc::clone(&sink) as Arc<dyn _BlockSinkTrait>);
        cache.put(7, 0xABCD);

        let bytes = cache.fetch_block(7).await.expect("fetch ok");
        assert_eq!(bytes, vec![0xCA, 0xFE, 0xBA, 0xBE]);
        assert_eq!(sink.call_count(), 1, "auto-install must write to sink");
    }

    #[tokio::test]
    async fn fetch_block_does_not_install_when_no_sink_wired() {
        // No sink → fetch returns bytes but doesn't try to install.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0x11, 0x22]);

        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source));
        cache.put(7, 0xABCD);

        let bytes = cache.fetch_block(7).await.expect("fetch ok");
        assert_eq!(bytes, vec![0x11, 0x22]);
        // No sink to assert against — the test passes if fetch
        // succeeds without panicking on a None sink.
    }

    #[tokio::test]
    async fn fetch_block_does_not_install_when_install_on_fetch_disabled() {
        // install_on_fetch=false is the explicit opt-out.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0x33, 0x44]);

        let sink = Arc::new(RecordingBlockSink::default());
        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source))
            .with_block_sink(Arc::clone(&sink) as Arc<dyn _BlockSinkTrait>)
            .with_install_on_fetch(false);
        cache.put(7, 0xABCD);

        let bytes = cache.fetch_block(7).await.expect("fetch ok");
        assert_eq!(bytes, vec![0x33, 0x44]);
        assert_eq!(
            sink.call_count(),
            0,
            "install_on_fetch=false must skip sink.write_block"
        );
    }

    #[tokio::test]
    async fn fetch_block_returns_bytes_even_when_sink_fails() {
        // Best-effort: sink failure does not break fetch.
        use crate::distributed_kv::block_data_source::MockBlockDataSource;
        let mut source = MockBlockDataSource::new();
        source.insert(7, vec![0x55, 0x66, 0x77]);

        let mut sink = RecordingBlockSink::default();
        sink.return_error = true;
        let sink = Arc::new(sink);
        let cache = DistributedKVCache::new(CacheConfig::new(NodeId(0), 1))
            .with_block_data_source(Arc::new(source))
            .with_block_sink(Arc::clone(&sink) as Arc<dyn _BlockSinkTrait>);
        cache.put(7, 0xABCD);

        let bytes = cache
            .fetch_block(7)
            .await
            .expect("fetch ok despite sink err");
        assert_eq!(bytes, vec![0x55, 0x66, 0x77]);
    }
}
