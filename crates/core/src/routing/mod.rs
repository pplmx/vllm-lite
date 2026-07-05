//! Request routing namespace: currently just [`hash_router`] for consistent-hash prefix routing. Round-robin / latency-based strategies slot in here.
#![allow(unused_imports)]
#![allow(dead_code)]

pub mod hash_router;

pub use hash_router::{HashRouter, NodeInfo};
