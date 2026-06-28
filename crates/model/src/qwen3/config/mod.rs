//! qwen3_config: qwen3 config.

mod model;
mod rope;

pub use model::{AttentionType, Qwen3Config, TextConfig};
pub use rope::{RopeParameters, RopeScaling, RopeType};
