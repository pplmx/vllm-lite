//! Hash-based request router: hash the prompt prefix to a backend node id, with consistent hashing so the same prefix lands on the same node across rebalances.
//!
//! Activated only when `vllm-core` is built with `--features routing`.
//! Single-node deployments instantiate a `HashRouter` with one node and
//! always route locally.
#![allow(dead_code)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

/// `NodeInfo`. See the type definition for fields and behavior.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Stable identifier of the backend node.
    pub node_id: String,
    /// Current load as reported by the node (interpretation is caller-defined; typically a 0.0–1.0 fraction).
    pub load: f64,
    /// Whether this node already has a cached prefix for the routing key.
    pub has_cache: bool,
}

/// Consistent-hash request router. Maps request keys to replicas using a hash ring; supports virtual nodes for smoother load distribution.
pub struct HashRouter {
    /// Pool of backend nodes available for routing.
    nodes: Arc<RwLock<Vec<NodeInfo>>>,
    #[allow(dead_code)]
    // audited 2026-06-26 (Wave 1): consistent-hashing virtual-nodes placeholder; current route() uses nodes.len() modulo
    /// Virtual-node multiplier (currently unused; reserved for hash-ring implementation).
    virtual_nodes: usize,
}

impl HashRouter {
    /// Create an empty router; `virtual_nodes` is reserved for future hash-ring use.
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            virtual_nodes,
        }
    }

    /// Register a backend node if its id is not already present.
    pub async fn add_node(&self, node_id: String) {
        let mut nodes = self.nodes.write().await;

        if !nodes.iter().any(|n| n.node_id == node_id) {
            nodes.push(NodeInfo {
                node_id,
                load: 0.0,
                has_cache: false,
            });
        }
    }

    /// Remove a backend node from the routing pool.
    pub async fn remove_node(&self, node_id: &str) {
        let mut nodes = self.nodes.write().await;
        nodes.retain(|n| n.node_id != node_id);
    }

    /// Update the reported load fraction for `node_id`.
    pub async fn update_node_load(&self, node_id: &str, load: f64) {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.iter_mut().find(|n| n.node_id == node_id) {
            node.load = load;
        }
    }

    /// Mark whether `node_id` already holds a prefix-cache entry for the key.
    pub async fn update_cache_status(&self, node_id: &str, has_cache: bool) {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.iter_mut().find(|n| n.node_id == node_id) {
            node.has_cache = has_cache;
        }
    }

    /// Route `key` to a node preferring cache hits, else least-loaded.
    pub async fn route(&self, key: &str) -> Option<String> {
        let nodes = self.nodes.read().await;

        if nodes.is_empty() {
            return None;
        }

        let hash = Self::hash_key(key);

        // Start from the hashed index and probe subsequent nodes (wrapping)
        // so a cache hit on any node is preferred over least-loaded fallback.
        // invariant: hash % nodes.len() fits in usize on all targets since
        // the modulus is bounded by the node count.
        #[allow(clippy::cast_possible_truncation)]
        let start_idx = (hash % nodes.len() as u64) as usize;
        for offset in 0..nodes.len() {
            let idx = (start_idx + offset) % nodes.len();
            let node = &nodes[idx];

            if node.has_cache {
                debug!(key = %key, node = %node.node_id, "Routing to node with cache");
                return Some(node.node_id.clone());
            }
        }

        let least_loaded = nodes.iter().min_by(|a, b| {
            a.load
                .partial_cmp(&b.load)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let result = least_loaded.map(|node| {
            debug!(key = %key, node = %node.node_id, load = node.load, "Routing to least loaded node");
            node.node_id.clone()
        });
        drop(nodes);
        result
    }

    /// Route by a precomputed prompt hash (consistent modulo node count).
    pub async fn route_by_prompt_hash(&self, prompt_hash: u64) -> Option<String> {
        let nodes = self.nodes.read().await;

        if nodes.is_empty() {
            return None;
        }

        // invariant: prompt_hash % nodes.len() fits in usize on all targets
        // since the modulus is bounded by the node count.
        #[allow(clippy::cast_possible_truncation)]
        let idx = (prompt_hash % nodes.len() as u64) as usize;
        let node = &nodes[idx];

        debug!(prompt_hash = prompt_hash, node = %node.node_id, "Routing by prompt hash");
        let result = Some(node.node_id.clone());
        drop(nodes);
        result
    }

    fn hash_key(key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Number of registered backend nodes.
    pub async fn get_node_count(&self) -> usize {
        self.nodes.read().await.len()
    }

    /// Snapshot of all registered node ids.
    pub async fn get_all_nodes(&self) -> Vec<String> {
        self.nodes
            .read()
            .await
            .iter()
            .map(|n| n.node_id.clone())
            .collect()
    }
}

impl Default for HashRouter {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_remove_nodes() {
        let router = HashRouter::default();

        router.add_node("node-0".to_string()).await;
        router.add_node("node-1".to_string()).await;

        assert_eq!(router.get_node_count().await, 2);

        router.remove_node("node-0").await;
        assert_eq!(router.get_node_count().await, 1);
    }

    #[tokio::test]
    async fn test_consistent_routing() {
        let router = HashRouter::default();

        router.add_node("node-0".to_string()).await;
        router.add_node("node-1".to_string()).await;
        router.add_node("node-2".to_string()).await;

        let key = "test-prompt-hash-12345";
        let first_route = router.route(key).await;
        let second_route = router.route(key).await;

        assert_eq!(first_route, second_route);
    }

    #[tokio::test]
    async fn test_route_by_prompt_hash() {
        let router = HashRouter::default();

        router.add_node("node-0".to_string()).await;
        router.add_node("node-1".to_string()).await;

        let hash = 12345u64;
        let route = router.route_by_prompt_hash(hash).await;

        assert!(route.is_some());
    }

    /// `update_node_load` / `remove_node` / `get_all_nodes` must do their job.
    ///
    /// `cargo mutants` replaced `update_node_load` with `()`, flipped both of
    /// its id comparisons (`==` -> `!=`) and its `remove_node` sibling
    /// (`!=` -> `==`, which keeps ONLY the removed node), and replaced
    /// `get_all_nodes` with `vec![]` and the placeholder lists — twelve
    /// survivors in a file whose only consumer is its own test module. The
    /// load value is the least-loaded fallback's input, and a `get_all_nodes`
    /// that returns nothing makes a router look empty to any inspector.
    #[tokio::test]
    async fn node_bookkeeping_updates_and_reads_back() {
        let router = HashRouter::new(100);
        assert!(router.get_all_nodes().await.is_empty());
        assert!(router.route("k").await.is_none(), "no nodes means no route");

        router.add_node("a".to_string()).await;
        router.add_node("b".to_string()).await;
        router.add_node("a".to_string()).await; // duplicate is ignored
        assert_eq!(router.get_node_count().await, 2);
        let mut ids = router.get_all_nodes().await;
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);

        // Load decides the fallback: make "b" the least loaded and expect it.
        router.update_node_load("a", 0.9).await;
        router.update_node_load("b", 0.1).await;
        for key in ["x", "y", "z", "w"] {
            assert_eq!(
                router.route(key).await,
                Some("b".to_string()),
                "the least-loaded node must win key {key}"
            );
        }

        // Swapping the loads swaps the winner, proving the update is real.
        router.update_node_load("a", 0.05).await;
        for key in ["x", "y", "z", "w"] {
            assert_eq!(router.route(key).await, Some("a".to_string()));
        }

        // An unknown id is a no-op, not a panic.
        router.update_node_load("missing", 0.0).await;

        // Removing "a" leaves exactly "b", and does NOT keep only "a".
        router.remove_node("a").await;
        assert_eq!(router.get_node_count().await, 1);
        assert_eq!(router.get_all_nodes().await, vec!["b".to_string()]);
        router.remove_node("b").await;
        assert_eq!(router.get_node_count().await, 0);
        assert!(router.route("k").await.is_none());
    }

    /// `route_by_prompt_hash` must select `hash % node_count`.
    ///
    /// `cargo mutants` replaced the method with `Some("")` / `Some("xyzzy")`
    /// and the `%` with `/` and `+`. The hash is the only thing that makes the
    /// mapping CONSISTENT across processes, so a division or a sum reshuffles
    /// every key to a different node — the prefix cache then misses on every
    /// peer. The fixture pins the exact index for a range of hashes.
    #[tokio::test]
    async fn route_by_prompt_hash_is_modulo_the_node_count() {
        let router = HashRouter::new(100);
        assert!(router.route_by_prompt_hash(0).await.is_none());

        router.add_node("n0".to_string()).await;
        router.add_node("n1".to_string()).await;
        router.add_node("n2".to_string()).await;

        for hash in 0u64..12 {
            let expected = format!("n{}", hash % 3);
            assert_eq!(
                router.route_by_prompt_hash(hash).await,
                Some(expected),
                "hash {hash} must land on node {}",
                hash % 3
            );
        }
        // A large hash must not panic or overflow the index.
        assert!(router.route_by_prompt_hash(u64::MAX).await.is_some());
    }

    /// `hash_key` must be a real hash: stable per key, distinct across keys.
    ///
    /// Replaced by the constants 0 and 1 it survived, because consistency was
    /// only ever checked by routing the SAME key twice — a constant is
    /// perfectly consistent. It is not, however, a hash: every key would land
    /// on node 0, so all prefixes pile onto one backend.
    #[test]
    fn hash_key_is_stable_and_discriminating() {
        let a = HashRouter::hash_key("prompt-a");
        let b = HashRouter::hash_key("prompt-a");
        let c = HashRouter::hash_key("prompt-b");
        assert_eq!(a, b, "the same key must hash the same");
        assert_ne!(a, c, "different keys must (almost surely) differ");
        assert_ne!(a, 0, "a constant-zero hash would pin every key to node 0");
    }

    /// The cache-hit probe must START at `hash % node_count`.
    ///
    /// `route` probes `start, start+1, ...` (wrapping) and returns the first
    /// node holding the prefix, so the start index decides WHICH cached node
    /// wins when several hold one. `cargo mutants` relaxed `hash % len` to
    /// `hash / len` and to `hash + len`; the first reshuffles the probe order,
    /// the second leaves `(start + offset) % len` unchanged for every query.
    ///
    /// The fixture marks two of three nodes as cache holders and searches for a
    /// key whose `hash % 3` and `hash / 3` start the walk at different cached
    /// nodes, so the assertion is about the ORDER, not about cache preference
    /// in general.
    #[tokio::test]
    async fn cache_probe_starts_at_hash_modulo_node_count() {
        let router = HashRouter::new(100);
        router.add_node("n0".to_string()).await;
        router.add_node("n1".to_string()).await;
        router.add_node("n2".to_string()).await;
        // Only n0 and n2 hold the prefix, so the walk order decides.
        router.update_cache_status("n0", true).await;
        router.update_cache_status("n2", true).await;

        let key = (0..10_000)
            .map(|i| format!("probe-key-{i}"))
            .find(|k| {
                let h = HashRouter::hash_key(k);
                h % 3 == 0 && (h / 3) % 3 == 1
            })
            .expect("some key starts the real and the `/` walk on different nodes");

        assert_eq!(
            router.route(&key).await,
            Some("n0".to_string()),
            "the walk starts at hash % 3 == 0, so n0 is reached before n2"
        );
    }

    #[tokio::test]
    async fn test_route_prefers_cache_hit_on_any_node() {
        let router = HashRouter::default();

        router.add_node("node-0".to_string()).await;
        router.add_node("node-1".to_string()).await;
        router.add_node("node-2".to_string()).await;

        // Mark node-1 as having a cache.
        router.update_cache_status("node-1", true).await;

        // Route any key — a cache hit should be returned, even if the
        // hash maps to a different node.
        let route = router.route("test-key").await;
        assert_eq!(route, Some("node-1".to_string()));
    }
}
