#![allow(clippy::module_inception)]

pub mod pipeline;
pub mod stage;

pub use pipeline::PipelineParallel;
pub use stage::{PipelineStage, PipelineStageConfig, StageInput, StageOutput};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("stage {stage} not ready")]
    StageNotReady { stage: usize },

    #[error("invalid stage count: {count}")]
    InvalidStageCount { count: usize },

    #[error("forward failed: {0}")]
    ForwardFailed(String),

    #[error("inter-stage transfer failed: {0}")]
    TransferFailed(String),

    #[error("device error: {0}")]
    DeviceError(String),
}

pub type Result<T> = std::result::Result<T, PipelineError>;

pub use stage::PipelineStage as PipelineStageTrait;
