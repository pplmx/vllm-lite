use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub load: f64,
    pub has_cache: bool,
}

pub struct HashRouter {
    nodes: Arc<RwLock<Vec<NodeInfo>>>,
    #[allow(dead_code)]
    virtual_nodes: usize,
}

impl HashRouter {
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            virtual_nodes,
        }
    }

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

    pub async fn remove_node(&self, node_id: &str) {
        let mut nodes = self.nodes.write().await;
        nodes.retain(|n| n.node_id != node_id);
    }

    pub async fn update_node_load(&self, node_id: &str, load: f64) {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.iter_mut().find(|n| n.node_id == node_id) {
            node.load = load;
        }
    }

    pub async fn update_cache_status(&self, node_id: &str, has_cache: bool) {
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.iter_mut().find(|n| n.node_id == node_id) {
            node.has_cache = has_cache;
        }
    }

    pub async fn route(&self, key: &str) -> Option<String> {
        let nodes = self.nodes.read().await;

        if nodes.is_empty() {
            return None;
        }

        let hash = self.hash_key(key);

        for _i in 0..nodes.len() {
            let idx = (hash % nodes.len() as u64) as usize;
            let node = &nodes[idx];

            if node.has_cache {
                debug!(key = %key, node = %node.node_id, "Routing to node with cache");
                return Some(node.node_id.clone());
            }
        }

        let least_loaded = nodes
            .iter()
            .min_by(|a, b| a.load.partial_cmp(&b.load).unwrap_or(std::cmp::Ordering::Equal));

        match least_loaded {
            Some(node) => {
                debug!(key = %key, node = %node.node_id, load = node.load, "Routing to least loaded node");
                Some(node.node_id.clone())
            }
            None => None,
        }
    }

    pub async fn route_by_prompt_hash(&self, prompt_hash: u64) -> Option<String> {
        let nodes = self.nodes.read().await;

        if nodes.is_empty() {
            return None;
        }

        let idx = (prompt_hash % nodes.len() as u64) as usize;
        let node = &nodes[idx];

        debug!(prompt_hash = prompt_hash, node = %node.node_id, "Routing by prompt hash");
        Some(node.node_id.clone())
    }

    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    pub async fn get_node_count(&self) -> usize {
        self.nodes.read().await.len()
    }

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
        let route1 = router.route(key).await;
        let route2 = router.route(key).await;

        assert_eq!(route1, route2);
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
}