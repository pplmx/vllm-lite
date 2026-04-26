use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use vllm_traits::SeqId;

use super::leader_election::LeaderElection;

#[derive(Debug, Clone)]
pub struct InFlightRequest {
    pub seq_id: SeqId,
    pub prompt_hash: u64,
    pub created_at: std::time::Instant,
    pub node_id: String,
}

pub struct FailoverManager {
    leader_election: Arc<LeaderElection>,
    inflight_requests: Arc<RwLock<HashMap<SeqId, InFlightRequest>>>,
    request_timeout: std::time::Duration,
}

impl FailoverManager {
    pub fn new(request_timeout: std::time::Duration) -> Self {
        Self {
            leader_election: Arc::new(LeaderElection::new()),
            inflight_requests: Arc::new(RwLock::new(HashMap::new())),
            request_timeout,
        }
    }

    pub fn leader_election(&self) -> Arc<LeaderElection> {
        self.leader_election.clone()
    }

    pub async fn track_request(&self, seq_id: SeqId, prompt_hash: u64, node_id: String) {
        let request = InFlightRequest {
            seq_id,
            prompt_hash,
            created_at: std::time::Instant::now(),
            node_id,
        };

        let mut requests = self.inflight_requests.write().await;
        requests.insert(seq_id, request);

        info!(seq_id = %seq_id, "Tracking in-flight request");
    }

    pub async fn complete_request(&self, seq_id: SeqId) {
        let mut requests = self.inflight_requests.write().await;
        if requests.remove(&seq_id).is_some() {
            info!(seq_id = %seq_id, "Request completed, removed from tracking");
        }
    }

    pub async fn get_stale_requests(&self) -> Vec<SeqId> {
        let mut requests = self.inflight_requests.write().await;
        let now = std::time::Instant::now();

        let stale: Vec<SeqId> = requests
            .iter()
            .filter(|(_, req)| now.duration_since(req.created_at) > self.request_timeout)
            .map(|(&seq_id, _)| seq_id)
            .collect();

        for seq_id in &stale {
            warn!(seq_id = %seq_id, "Request timed out");
            requests.remove(seq_id);
        }

        stale
    }

    pub async fn get_inflight_count(&self) -> usize {
        self.inflight_requests.read().await.len()
    }

    pub async fn migrate_request(&self, seq_id: SeqId, new_node: String) -> bool {
        let mut requests = self.inflight_requests.write().await;

        if let Some(request) = requests.get_mut(&seq_id) {
            info!(seq_id = %seq_id, new_node = %new_node, "Migrating request to new node");
            request.node_id = new_node;
            return true;
        }

        false
    }

    pub async fn on_node_failure(&self, failed_node: &str) -> Vec<SeqId> {
        let mut requests = self.inflight_requests.write().await;

        let to_migrate: Vec<SeqId> = requests
            .iter()
            .filter(|(_, req)| req.node_id == failed_node)
            .map(|(&seq_id, _)| seq_id)
            .collect();

        info!(
            failed_node = %failed_node,
            count = to_migrate.len(),
            "Node failed, identified requests to migrate"
        );

        to_migrate
    }

    pub async fn handle_leader_transfer(&self, new_leader: String) {
        let is_current_leader = self.leader_election.is_leader().await;

        if is_current_leader {
            info!(new_leader = %new_leader, "Transferring leadership, preserving in-flight requests");
            self.leader_election.step_down().await;
        }

        self.leader_election.on_leader_lost(Some(new_leader)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_track_request() {
        let manager = FailoverManager::new(std::time::Duration::from_secs(60));

        manager.track_request(1, 12345, "node-0".to_string()).await;

        assert_eq!(manager.get_inflight_count().await, 1);
    }

    #[tokio::test]
    async fn test_complete_request() {
        let manager = FailoverManager::new(std::time::Duration::from_secs(60));

        manager.track_request(1, 12345, "node-0".to_string()).await;
        manager.complete_request(1).await;

        assert_eq!(manager.get_inflight_count().await, 0);
    }

    #[tokio::test]
    async fn test_node_failure() {
        let manager = FailoverManager::new(std::time::Duration::from_secs(60));

        manager.track_request(1, 12345, "node-0".to_string()).await;
        manager.track_request(2, 12346, "node-1".to_string()).await;

        let failed = manager.on_node_failure("node-0").await;
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0], 1);
    }
}