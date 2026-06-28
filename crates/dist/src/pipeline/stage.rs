use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

/// PipelineStageConfig: pipeline stage configuration.
#[derive(Debug, Clone)]
pub struct PipelineStageConfig {
    pub stage_id: usize,
    pub num_stages: usize,
    pub num_layers: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub device: Device,
}

impl PipelineStageConfig {
    pub fn new(stage_id: usize, num_stages: usize, num_layers: usize, device: Device) -> Self {
        let layers_per_stage = num_layers.div_ceil(num_stages);
        let layer_start = stage_id * layers_per_stage;
        let layer_end = (layer_start + layers_per_stage).min(num_layers);

        Self {
            stage_id,
            num_stages,
            num_layers,
            layer_start,
            layer_end,
            device,
        }
    }

    pub fn num_layers_in_stage(&self) -> usize {
        self.layer_end - self.layer_start
    }

    pub fn is_first_stage(&self) -> bool {
        self.stage_id == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.stage_id == self.num_stages - 1
    }
}

/// StageInput: stage input.
#[derive(Debug, Clone)]
pub struct StageInput {
    pub hidden_states: Tensor,
    pub position_ids: Tensor,
    pub positions: Vec<usize>,
    pub kv_block_ids: Vec<Vec<usize>>,
}

/// StageOutput: stage output.
#[derive(Debug, Clone)]
pub struct StageOutput {
    pub hidden_states: Tensor,
    pub position_ids: Tensor,
    pub next_positions: Vec<usize>,
    pub kv_block_ids: Vec<Vec<usize>>,
    pub is_generating: bool,
}

/// PipelineStage: pipeline stage trait.
pub trait PipelineStage: Send + Sync {
    fn config(&self) -> &PipelineStageConfig;

    fn forward(&self, input: StageInput) -> Result<StageOutput>;

    fn forward_microbatches(&self, inputs: Vec<StageInput>) -> Result<Vec<StageOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.forward(input)?);
        }
        Ok(outputs)
    }
}

/// Default no-op `PipelineStage`.
///
/// Wraps a single-stage CPU config. `forward` always errors — the stage
/// holds no weights and cannot produce real outputs. Useful as a placeholder
/// when pipeline parallelism is disabled.
#[derive(Debug)]
pub(crate) struct NoopPipelineStage {
    config: PipelineStageConfig,
}

impl Default for NoopPipelineStage {
    fn default() -> Self {
        Self {
            config: PipelineStageConfig::new(0, 1, 1, Device::Cpu),
        }
    }
}

impl PipelineStage for NoopPipelineStage {
    fn config(&self) -> &PipelineStageConfig {
        &self.config
    }

    fn forward(&self, _input: StageInput) -> Result<StageOutput> {
        Err(candle_core::Error::msg(
            "NoopPipelineStage: no pipeline parallelism configured",
        ))
    }
}

impl dyn PipelineStage {
    /// Returns an `Arc<Self>` wrapping the no-op `NoopPipelineStage`.
    ///
    /// This is the closest equivalent to
    /// `Arc::<dyn PipelineStage>::default()`; Rust's orphan rule prevents
    /// a direct `impl Default for Arc<dyn ...>` because `Arc` is foreign and
    /// there is no local type appearing before the uncovered trait-object
    /// parameter.
    pub fn default_arc() -> Arc<Self> {
        Arc::new(NoopPipelineStage::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config_equal_layers() {
        let config = PipelineStageConfig::new(0, 2, 4, Device::Cpu);

        assert_eq!(config.stage_id, 0);
        assert_eq!(config.num_stages, 2);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 2);
        assert_eq!(config.num_layers_in_stage(), 2);
    }

    #[test]
    fn test_stage_config_unequal_layers() {
        let config = PipelineStageConfig::new(1, 2, 5, Device::Cpu);

        assert_eq!(config.layer_start, 3); // stage 1 starts at layer 3 (ceil(5/2) = 3)
        assert_eq!(config.layer_end, 5);
        assert_eq!(config.num_layers_in_stage(), 2); // layers 3 and 4 = 2 layers
    }

    #[test]
    fn test_stage_first_last() {
        let first = PipelineStageConfig::new(0, 3, 9, Device::Cpu);
        let middle = PipelineStageConfig::new(1, 3, 9, Device::Cpu);
        let last = PipelineStageConfig::new(2, 3, 9, Device::Cpu);

        assert!(first.is_first_stage());
        assert!(!first.is_last_stage());

        assert!(!middle.is_first_stage());
        assert!(!middle.is_last_stage());

        assert!(!last.is_first_stage());
        assert!(last.is_last_stage());
    }

    #[test]
    fn pipeline_stage_default_arc_is_noop() {
        let stage: Arc<dyn PipelineStage> = <dyn PipelineStage>::default_arc();
        assert_eq!(stage.config().stage_id, 0);
        assert_eq!(stage.config().num_stages, 1);
    }
}
