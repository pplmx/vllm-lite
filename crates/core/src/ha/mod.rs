//! High-availability namespace: leader election + failover.
//!
//! Active in multi-replica deployments only; in single-node setups
//! the local node is trivially the leader.

pub mod failover;
pub mod leader_election;
