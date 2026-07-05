//! Radix-tree prefix cache: tree of [`RadixNode`]s ([`tree::RadixTree`]) shared across requests so identical prompt prefixes reuse KV blocks.
//!
//! `PrefixMatchResult` (in `tree.rs`) carries the per-request lookup
//! outcome: matched-token count, child block-ids, and the optional
//! parent-link hash chain.
pub mod node;
pub mod tree;
pub use node::RadixNode;
pub use tree::{PrefixMatchResult, RadixTree};
