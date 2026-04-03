//! Gemma4 Model implementation.

#![allow(dead_code)]

use crate::config::ModelConfig;
use crate::gemma4::block::Gemma4Block;
use candle_core::{Device, Result as CandleResult};
use candle_nn::VarBuilder;
use vllm_traits::{BatchOutput, ModelBackend, Result as EngineResult, SeqId, TokenId};

pub struct Gemma4Model {
    config: ModelConfig,
    _layers: Vec<Gemma4Block>,
    device: Device,
}

impl Gemma4Model {
    pub fn new(config: ModelConfig, device: Device, _num_kv_blocks: usize) -> CandleResult<Self> {
        let num_layers = config.num_layers;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = Gemma4Block::new(&config, i, vb.pp(format!("layers.{}", i)))?;
            layers.push(layer);
        }

        Ok(Self {
            config,
            _layers: layers,
            device,
        })
    }
}

impl ModelBackend for Gemma4Model {
    fn forward(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<BatchOutput> {
        todo!()
    }

    fn forward_logits(
        &mut self,
        _seq_ids: &[SeqId],
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
        _kv_block_ids: &[Vec<usize>],
        _num_computed_tokens: &[usize],
        _is_prefill: &[bool],
    ) -> EngineResult<Vec<Vec<f32>>> {
        todo!()
    }

    fn embed(
        &mut self,
        _input_tokens: &[Vec<TokenId>],
        _positions: &[Vec<usize>],
    ) -> EngineResult<Vec<Vec<f32>>> {
        todo!()
    }
}
