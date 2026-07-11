//! Radix-tree node: one vertex in the prefix-cache tree, holding a token range + child edges keyed by next-token id.
//!
//! Trees are composed by `radix_cache/tree.rs`; this file owns only
//! the per-node storage and child-lookup helpers.
#![allow(clippy::module_name_repetitions)]
use std::collections::HashMap;
use std::sync::Arc;
use vllm_traits::{BlockId, TokenId};

#[derive(Debug)]
/// Radix Tree node
pub struct RadixNode {
    /// Token sequence for this node
    pub tokens: Vec<TokenId>,
    /// KV blocks if this is a complete entry
    pub blocks: Option<Arc<Vec<BlockId>>>,
    /// Child nodes: key is next token
    pub children: HashMap<TokenId, Box<Self>>,
    /// Is this a complete cache entry
    pub is_complete: bool,
}

impl RadixNode {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            blocks: None,
            children: HashMap::new(),
            is_complete: false,
        }
    }
}

impl Default for RadixNode {
    fn default() -> Self {
        Self::new()
    }
}
