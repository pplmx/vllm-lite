use std::collections::HashMap;
use std::sync::Arc;
use vllm_traits::{BlockId, TokenId};

/// Radix Tree node
pub struct RadixNode {
    /// Token sequence for this node
    pub tokens: Vec<TokenId>,
    /// KV blocks if this is a complete entry
    pub blocks: Option<Arc<Vec<BlockId>>>,
    /// Child nodes: key is next token
    pub children: HashMap<TokenId, Box<RadixNode>>,
    /// Is this a complete cache entry
    pub is_complete: bool,
}

impl RadixNode {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            blocks: None,
            children: HashMap::new(),
            is_complete: false,
        }
    }

    pub fn with_tokens(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
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
