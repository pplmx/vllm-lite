//! High-availability namespace: leader election + failover.
//!
//! Active in multi-replica deployments only; in single-node setups
//! the local node is trivially the leader.
#![allow(unused_imports)]
#![allow(dead_code)]

pub mod failover;
pub mod leader_election;

pub use failover::FailoverManager;
pub use leader_election::{LeaderElection, LeadershipState};
