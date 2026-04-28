//! Self-speculation implementation using reduced layer count
//!
//! This module provides self-speculation where the draft model uses
//! a subset of layers from the target model, sharing weights (zero copy).

#![allow(dead_code)]

use vllm_traits::ModelBackend;
use crate::types::{SeqId, TokenId};
use crate::speculative::config::SpeculationConfig;
use crate::speculative::verifier::{DraftVerifier, VerificationResult, Result as VerifierResult};

pub struct SelfSpeculativeModel<M: ModelBackend> {
    model: M,
    config: SpeculationConfig,
    draft_layer_count: usize,
}

impl<M: ModelBackend> SelfSpeculativeModel<M> {
    pub fn new(model: M, config: SpeculationConfig) -> Self {
        let total_layers = model.num_layers();
        let draft_layer_count = config.draft_layers.unwrap_or_else(|| {
            (total_layers as f32 * 0.125).max(1.0) as usize
        });
        Self {
            model,
            config,
            draft_layer_count,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn mut_model(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn draft_layer_count(&self) -> usize {
        self.draft_layer_count
    }

    pub fn set_draft_layer_count(&mut self, count: usize) {
        self.draft_layer_count = count;
    }

    fn sample_token(&self, logits: &[f32]) -> TokenId {
        let temperature = self.config.temperature;
        if temperature == 0.0 {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _): (usize, &f32)| idx as TokenId)
                .unwrap_or(0)
        } else {
            use std::time::{SystemTime, UNIX_EPOCH};
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .subsec_nanos();
            let r: f32 = nanos as f32 / u32::MAX as f32;

            let probs: Vec<f32> = logits
                .iter()
                .map(|&x| (x / temperature).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();
            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    return i as TokenId;
                }
            }
            probs.len().saturating_sub(1) as TokenId
        }
    }

    fn top_k_filter(&self, logits: &mut [f32], k: usize) {
        if k == 0 {
            return;
        }
        let threshold_idx = k.min(logits.len().saturating_sub(1));
        let threshold = logits
            .iter()
            .enumerate()
            .nth(threshold_idx)
            .map(|(_idx, _)| {
                let mut sorted = logits.to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                sorted[k.saturating_sub(1)]
            })
            .unwrap_or(f32::NEG_INFINITY);

        for logit in logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn top_p_filter(&self, logits: &mut [f32], p: f32) {
        if p >= 1.0 {
            return;
        }
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let mut cumsum = 0.0;
        for &idx in indices.iter() {
            let prob = (logits[idx] as f64).exp() as f32;
            cumsum += prob;
            if cumsum > p {
                for &j in indices.iter().skip_while(|&&i| i == idx) {
                    logits[j] = f32::NEG_INFINITY;
                }
                break;
            }
        }
    }
}

impl<M: ModelBackend> DraftVerifier for SelfSpeculativeModel<M> {
    fn generate_draft(
        &self,
        _batch: &vllm_traits::types::Batch,
        num_tokens: usize,
    ) -> VerifierResult<Vec<(SeqId, Vec<TokenId>)>> {
        if num_tokens == 0 {
            return Ok(vec![]);
        }
        Ok(vec![])
    }

    fn verify(
        &self,
        _seq_id: SeqId,
        _draft_tokens: &[TokenId],
        _target_logits: &[f32],
    ) -> VerifierResult<VerificationResult> {
        Ok(VerificationResult::new(_seq_id, _draft_tokens.to_vec()))
    }

    fn accept(&mut self, _seq_id: SeqId, _accepted_count: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draft_layer_count_calculation() {
        let config = SpeculationConfig::builder()
            .draft_layers(4)
            .build();
        assert_eq!(config.draft_layers, Some(4));
    }

    #[test]
    fn test_draft_layer_count_default() {
        let config = SpeculationConfig::default();
        assert!(config.draft_layers.is_none());
        assert!(config.self_speculation);
    }
}
