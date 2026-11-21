#![allow(dead_code, clippy::module_name_repetitions)]

//! High-availability failover: detect peer failure via health checks and promote a standby to active.
//!
//! Coordinates with the circuit breaker for failure detection. Activated
//! when `--features ha` is set in `vllm-core`; in single-node mode this
//! module is a no-op.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use vllm_traits::SeqId;

use super::leader_election::LeaderElection;

/// Request payload for `InFlight`. Contains input data, configuration, and request-tracking metadata.
#[derive(Debug, Clone)]
pub struct InFlightRequest {
    /// Sequence id of the tracked request.
    pub seq_id: SeqId,
    /// Hash of the prompt prefix (used for de-dup during failover).
    pub prompt_hash: u64,
    /// Instant when tracking started (basis for staleness check).
    pub created_at: std::time::Instant,
    /// Originating node id; updated by `migrate_request` on failover.
    pub node_id: String,
}

/// High-availability failover coordinator. Detects leader liveness via heartbeats and triggers leader election when the miss-threshold is crossed.
pub struct FailoverManager {
    /// Shared leader-election state (term-based, Raft-inspired).
    leader_election: Arc<LeaderElection>,
    /// Tracked in-flight requests keyed by sequence id.
    inflight_requests: Arc<RwLock<HashMap<SeqId, InFlightRequest>>>,
    /// How old a request can get before it is reported as stale.
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
        drop(requests);
    }

    pub async fn complete_request(&self, seq_id: SeqId) {
        let mut requests = self.inflight_requests.write().await;
        if requests.remove(&seq_id).is_some() {
            info!(seq_id = %seq_id, "Request completed, removed from tracking");
        }
        drop(requests);
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
            drop(requests);
            return true;
        }

        drop(requests);
        false
    }

    pub async fn on_node_failure(&self, failed_node: &str) -> Vec<SeqId> {
        // Read-only scan: a read lock keeps concurrent writers
        // (track/complete/migrate) unblocked on the failure path,
        // when many requests may be migrating at once.
        let requests = self.inflight_requests.read().await;

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

        drop(requests);
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

    /// `leader_election()` must hand out the manager's OWN election state.
    ///
    /// Returning a fresh `Arc::new(Default::default())` survived the suite,
    /// which is the worst kind of failure for this accessor: a caller reads a
    /// permanent "no leader / not leader" view while the manager's real state
    /// moves on, so health and routing decisions would be made on a ghost.
    #[tokio::test]
    async fn test_leader_election_accessor_shares_the_managers_state() {
        let manager = FailoverManager::new(std::time::Duration::from_secs(60));

        manager
            .leader_election()
            .become_leader("node-0".to_string())
            .await;

        assert!(
            manager.leader_election().is_leader().await,
            "leadership taken through the accessor must be visible through it"
        );
        assert!(
            manager.leader_election.is_leader().await,
            "the accessor must expose the manager's own state, not a new instance"
        );
    }

    /// Stale requests are exactly those past `request_timeout`, and they are
    /// RELEASED, not merely reported.
    ///
    /// `cargo mutants` flipped the `>` comparison four ways and emptied the
    /// removal loop, all surviving: nothing exercised the timeout path.
    #[tokio::test]
    async fn test_get_stale_requests_reports_and_releases() {
        // A zero timeout makes every tracked request immediately stale.
        let manager = FailoverManager::new(std::time::Duration::ZERO);
        manager.track_request(1, 12345, "node-0".to_string()).await;
        manager.track_request(2, 12346, "node-0".to_string()).await;

        let mut stale = manager.get_stale_requests().await;
        stale.sort_unstable();
        assert_eq!(
            stale,
            vec![1, 2],
            "every timed-out request must be reported"
        );
        assert_eq!(
            manager.get_inflight_count().await,
            0,
            "reported requests must also be released"
        );

        // A long timeout keeps a fresh request in flight.
        let patient = FailoverManager::new(std::time::Duration::from_secs(3600));
        patient.track_request(3, 12347, "node-0".to_string()).await;
        assert!(
            patient.get_stale_requests().await.is_empty(),
            "a request inside its timeout must not be reported as stale"
        );
        assert_eq!(patient.get_inflight_count().await, 1);
    }

    /// `migrate_request` must move a tracked request and report whether it
    /// found one.
    ///
    /// Replacing the whole function with `true` / `false` survived, so callers
    /// could not tell a successful migration from a miss.
    #[tokio::test]
    async fn test_migrate_request_reports_whether_it_found_the_request() {
        let manager = FailoverManager::new(std::time::Duration::from_secs(60));
        manager.track_request(7, 999, "node-0".to_string()).await;

        assert!(
            manager.migrate_request(7, "node-1".to_string()).await,
            "migrating a tracked request must succeed"
        );
        assert!(
            !manager.migrate_request(404, "node-1".to_string()).await,
            "migrating an unknown request must report failure"
        );
    }

    /// A leader transfer must demote this node and follow the new leader.
    #[tokio::test]
    async fn test_handle_leader_transfer_demotes_and_follows() {
        let manager = FailoverManager::new(std::time::Duration::from_secs(60));
        manager
            .leader_election
            .become_leader("node-0".to_string())
            .await;
        assert!(manager.leader_election.is_leader().await);

        manager.handle_leader_transfer("node-1".to_string()).await;

        assert!(
            !manager.leader_election.is_leader().await,
            "a transferred leadership must demote the previous leader"
        );
        assert_eq!(
            manager.leader_election.get_leader_id().await,
            Some("node-1".to_string()),
            "the new leader must be recorded"
        );
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
