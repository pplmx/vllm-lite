//! qwen3: qwen3 module.

/// arch: arch module.
pub mod arch;
/// block: block module.
pub mod block;
/// config: config module.
pub mod config;
/// mla_attention: mla attention module.
pub mod mla_attention;
/// model: model module.
pub mod model;
/// register: register module.
pub mod register;

/// tp: tp module.
#[cfg(feature = "multi-node")]
pub mod tp;

pub use arch::Qwen3Architecture;
pub use config::{Qwen3Config, RopeParameters, RopeScaling, TextConfig};
pub use mla_attention::Qwen3MlaAttention;
pub use model::Qwen3Model;
