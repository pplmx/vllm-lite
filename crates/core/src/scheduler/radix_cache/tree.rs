#![allow(clippy::module_name_repetitions)]
use super::node::RadixNode;
use std::sync::Arc;
use vllm_traits::{BlockId, TokenId};

/// Prefix match result
#[derive(Clone, Debug)]
pub struct PrefixMatchResult {
    pub blocks: Arc<Vec<BlockId>>,
    pub matched_tokens: usize,
}

#[derive(Debug)]
/// Radix Tree for prefix caching
pub struct RadixTree {
    root: RadixNode,
    entry_count: usize,
}

impl RadixTree {
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(),
            entry_count: 0,
        }
    }

    /// Find longest prefix match - O(k) complexity.
    #[must_use]
    pub fn longest_prefix_match(&self, tokens: &[TokenId]) -> Option<PrefixMatchResult> {
        let mut node = &self.root;
        let mut matched_len = 0;
        let mut matched_blocks = None;

        for (i, &token) in tokens.iter().enumerate() {
            if let Some(child) = node.children.get(&token) {
                matched_len = i + 1;
                node = child;
                if node.is_complete {
                    matched_blocks.clone_from(&node.blocks);
                }
            } else {
                break;
            }
        }

        matched_blocks.map(|blocks| PrefixMatchResult {
            blocks,
            matched_tokens: matched_len,
        })
    }

    /// Insert new cache entry
    pub fn insert(&mut self, tokens: &[TokenId], blocks: Vec<BlockId>) {
        if tokens.is_empty() {
            return;
        }

        let mut node = &mut self.root;
        for &token in tokens {
            node = node
                .children
                .entry(token)
                .or_insert_with(|| Box::new(RadixNode::new()));
        }

        node.blocks = Some(Arc::new(blocks));
        node.is_complete = true;
        self.entry_count += 1;
    }

    /// Get entry count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entry_count
    }

    /// Check if tree is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.root.children.clear();
        self.entry_count = 0;
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_tree_insert_and_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2, 3], vec![10, 20, 30]);

        let result = tree.longest_prefix_match(&[1, 2, 3, 4, 5]);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.matched_tokens, 3);
        assert_eq!(result.blocks.as_ref(), &vec![10, 20, 30]);
    }

    #[test]
    fn test_radix_tree_no_match() {
        let tree = RadixTree::new();
        let result = tree.longest_prefix_match(&[1, 2, 3]);
        assert!(result.is_none());
    }

    #[test]
    fn test_radix_tree_partial_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2], vec![10, 20]);

        let result = tree.longest_prefix_match(&[1, 2, 3]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().matched_tokens, 2);
    }

    #[test]
    fn test_radix_tree_multiple_inserts() {
        let mut tree = RadixTree::new();
        tree.insert(&[1, 2], vec![10, 20]);
        tree.insert(&[1, 2, 3], vec![10, 20, 30]);
        tree.insert(&[4, 5], vec![40, 50]);

        let result = tree.longest_prefix_match(&[1, 2, 3, 4]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().matched_tokens, 3);

        let result = tree.longest_prefix_match(&[4, 5, 6]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().matched_tokens, 2);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate random token sequences (length 1..16).
    fn tokens_strategy() -> impl Strategy<Value = Vec<TokenId>> {
        proptest::collection::vec(any::<TokenId>(), 1..16)
    }

    /// Generate random block-id sequences (length 1..8).
    fn blocks_strategy() -> impl Strategy<Value = Vec<BlockId>> {
        proptest::collection::vec(any::<BlockId>(), 1..8)
    }

    proptest! {
        /// Insert + exact-match returns the inserted blocks.
        #[test]
        fn prop_radix_insert_then_match(
            key in tokens_strategy(),
            blocks in blocks_strategy(),
        ) {
            let mut tree = RadixTree::new();
            tree.insert(&key, blocks.clone());
            let result = tree.longest_prefix_match(&key);
            prop_assert!(result.is_some(), "exact-match lookup returned None after insert");
            // invariant: pre-validated result is unwrapped here.
            let result = result.unwrap();
            prop_assert_eq!(result.matched_tokens, key.len());
            prop_assert_eq!(result.blocks.as_ref(), &blocks);
        }

        /// Longest-prefix-match never claims more tokens than the query has.
        #[test]
        fn prop_longest_prefix_bounded(
            keys in proptest::collection::vec(tokens_strategy(), 0..4),
            query in tokens_strategy(),
        ) {
            let mut tree = RadixTree::new();
            for key in &keys {
                tree.insert(key, vec![1]);
            }
            if let Some(result) = tree.longest_prefix_match(&query) {
                prop_assert!(
                    result.matched_tokens <= query.len(),
                    "matched_tokens {} exceeds query length {}",
                    result.matched_tokens,
                    query.len()
                );
            }
        }

        /// Insert + clear + lookup returns None for any key.
        #[test]
        fn prop_insert_clear_returns_none(
            key in tokens_strategy(),
            blocks in blocks_strategy(),
        ) {
            let mut tree = RadixTree::new();
            tree.insert(&key, blocks);
            prop_assert!(!tree.is_empty());
            tree.clear();
            prop_assert!(tree.is_empty());
            prop_assert!(tree.longest_prefix_match(&key).is_none());
        }
    }
}
