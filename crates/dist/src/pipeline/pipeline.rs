#![allow(clippy::module_name_repetitions)]
use super::stage::{PipelineStage, StageInput, StageOutput};
use candle_core::Result;
use std::sync::Arc;

#[derive(Debug)]
/// `PipelineParallel`: pipeline parallel.
pub struct PipelineParallel {
    stages: Vec<Arc<dyn PipelineStage>>,
    config: PipelineParallelConfig,
}

/// `PipelineParallelConfig`: pipeline parallel configuration.
#[derive(Debug, Clone)]
pub struct PipelineParallelConfig {
    pub num_stages: usize,
    pub num_microbatches: usize,
    pub enable_async: bool,
    pub prefetch_ahead: usize,
}

impl Default for PipelineParallelConfig {
    fn default() -> Self {
        Self {
            num_stages: 1,
            num_microbatches: 4,
            enable_async: false,
            prefetch_ahead: 1,
        }
    }
}

impl PipelineParallel {
    #[must_use]
    pub fn new(config: PipelineParallelConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
        }
    }

    pub fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) {
        self.stages.push(stage);
    }

    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    pub const fn config(&self) -> &PipelineParallelConfig {
        &self.config
    }

    #[must_use]
    pub fn is_pipeline_parallel(&self) -> bool {
        self.stages.len() > 1
    }

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error).
    pub fn forward(&self, input: StageInput) -> Result<StageOutput> {
        if !self.is_pipeline_parallel() {
            if let Some(stage) = self.stages.first() {
                return stage.forward(input);
            }
            return Result::Err(candle_core::Error::msg("No stages available"));
        }

        let mut current_input = input;

        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let output = stage.forward(current_input)?;

            if stage_idx < self.stages.len() - 1 {
                current_input = StageInput {
                    hidden_states: output.hidden_states.clone(),
                    position_ids: output.position_ids.clone(),
                    positions: output.next_positions.clone(),
                    kv_block_ids: output.kv_block_ids.clone(),
                };
            } else {
                return Ok(output);
            }
        }

        Result::Err(candle_core::Error::msg("No output generated"))
    }

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_microbatches(&self, inputs: Vec<StageInput>) -> Result<Vec<StageOutput>> {
        if !self.is_pipeline_parallel() {
            if let Some(stage) = self.stages.first() {
                return stage.forward_microbatches(inputs);
            }
            return Result::Err(candle_core::Error::msg("No stages available"));
        }

        let mut all_outputs = Vec::with_capacity(inputs.len());
        let chunk_size = self.config.num_microbatches.max(1);

        for chunk in inputs.chunks(chunk_size) {
            for input in chunk {
                let output = self.forward(input.clone())?;
                all_outputs.push(output);
            }
        }

        Ok(all_outputs)
    }

    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn forward_with_schedule(&self, inputs: Vec<StageInput>) -> Result<Vec<StageOutput>> {
        if !self.is_pipeline_parallel() {
            return self.forward_microbatches(inputs);
        }

        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let output = self.forward(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }
}

impl Default for PipelineParallel {
    fn default() -> Self {
        Self::new(PipelineParallelConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::super::PipelineStageConfig;
    use super::*;
    use candle_core::{Device, Tensor};
    use std::sync::Mutex;

    #[derive(Debug)]
    struct MockStage {
        config: PipelineStageConfig,
        processed_count: Mutex<usize>,
    }

    impl MockStage {
        fn new(stage_id: usize, num_stages: usize) -> Self {
            Self {
                config: PipelineStageConfig::new(stage_id, num_stages, num_stages, Device::Cpu),
                processed_count: Mutex::new(0),
            }
        }
    }

    impl PipelineStage for MockStage {
        fn config(&self) -> &PipelineStageConfig {
            &self.config
        }

        fn forward(&self, input: StageInput) -> Result<StageOutput> {
            let mut count = self
                .processed_count
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *count += 1;

            Ok(StageOutput {
                hidden_states: input.hidden_states,
                position_ids: input.position_ids,
                next_positions: input.positions,
                kv_block_ids: input.kv_block_ids,
                is_generating: false,
            })
        }
    }

    #[test]
    fn test_single_stage() {
        let pipeline = PipelineParallel::default();
        assert!(!pipeline.is_pipeline_parallel());
    }

    #[test]
    fn test_multiple_stages() {
        let mut pipeline = PipelineParallel::new(PipelineParallelConfig {
            num_stages: 2,
            ..Default::default()
        });

        pipeline.add_stage(Arc::new(MockStage::new(0, 2)));
        pipeline.add_stage(Arc::new(MockStage::new(1, 2)));

        assert!(pipeline.is_pipeline_parallel());
        assert_eq!(pipeline.num_stages(), 2);
    }

    #[test]
    fn test_forward_through_stages() -> candle_core::Result<()> {
        let mut pipeline = PipelineParallel::default();

        for i in 0..3 {
            pipeline.add_stage(Arc::new(MockStage::new(i, 3)));
        }

        let input = StageInput {
            hidden_states: Tensor::ones((1, 10, 64), candle_core::DType::F32, &Device::Cpu)?,
            position_ids: Tensor::ones((1, 10), candle_core::DType::U32, &Device::Cpu)?,
            positions: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            kv_block_ids: vec![vec![0]],
        };

        let result = pipeline.forward(input);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_microbatches() {
        let mut pipeline = PipelineParallel::default();
        pipeline.add_stage(Arc::new(MockStage::new(0, 1)));

        let inputs: Vec<_> = (0..3)
            .map(|_| StageInput {
                hidden_states: Tensor::ones((1, 10, 64), candle_core::DType::F32, &Device::Cpu)
                    .unwrap(),
                position_ids: Tensor::ones((1, 10), candle_core::DType::U32, &Device::Cpu).unwrap(),
                positions: vec![0; 10],
                kv_block_ids: vec![vec![0]],
            })
            .collect();

        let outputs = pipeline.forward_microbatches(inputs);
        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap().len(), 3);
    }
}
