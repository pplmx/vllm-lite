//! MLP namespace: gated feed-forward implementations. Currently just `SwiGLU`; Gemma4 adds a gated `GeLU` variant.
/// SwiGLU feed-forward network (SiLU-gated linear unit).
pub mod swiglu;

pub use swiglu::{SwiGLU, swiglu_forward};
