//! Gemma4 Transformer Block implementation.

use crate::gemma4::attention::Gemma4Attention;
use crate::gemma4::mlp::GeGLU;
use crate::gemma4::rope::Gemma4RoPE;

/// Gemma4 transformer block with hybrid attention and GeGLU MLP.
pub struct Gemma4Block {
    _attention: Gemma4Attention,
    _mlp: GeGLU,
    _rope: Gemma4RoPE,
}

impl Gemma4Block {
    /// Create a new Gemma4Block.
    pub fn new() -> Self {
        Self {
            _attention: Gemma4Attention,
            _mlp: GeGLU,
            _rope: Gemma4RoPE,
        }
    }
}

impl Default for Gemma4Block {
    fn default() -> Self {
        Self::new()
    }
}
