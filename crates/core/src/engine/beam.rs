// Sub-module for beam search methods on Engine.
// See mod.rs for the Engine struct definition.

use crate::beam::BeamSequence;
use crate::engine::Engine;
use crate::error::Result;
use crate::sync::lock_mutex;
use vllm_traits::TokenId;

impl Engine {
    /// Runs the operation.
    /// # Errors
    ///
    /// Returns `Err` if the operation fails.
    pub fn step_beam(
        &mut self,
        beam_width: usize,
        length_penalty: f32,
        max_tokens: usize,
    ) -> Result<Vec<BeamSequence>> {
        let _batch = self.scheduler.build_batch();

        let mut results = Vec::new();
        for seq in self.scheduler.running() {
            let beam = self.beam_search(&seq, beam_width, length_penalty, max_tokens)?;
            results.push(beam);
        }

        Ok(results)
    }

    fn beam_search(
        &self,
        initial: &crate::types::Sequence,
        beam_width: usize,
        length_penalty: f32,
        max_tokens: usize,
    ) -> Result<BeamSequence> {
        let mut beams = vec![BeamSequence::new(
            initial.tokens.clone(),
            0.0,
            initial.kv_blocks.as_ref().clone(),
        )];

        for _ in 0..max_tokens {
            let mut all_candidates = Vec::new();

            for beam in &beams {
                if beam.tokens.is_empty() {
                    continue;
                }

                let logits = lock_mutex(&self.target_model)?.forward_logits(
                    &[0],
                    &[vec![beam.tokens.last().copied().unwrap_or(0)]],
                    &[vec![beam.tokens.len()]],
                    &[vec![beam.kv_blocks.last().copied().unwrap_or(0)]],
                    &[beam.tokens.len()],
                    &[false],
                )?;

                let top_k = Self::get_top_k(&logits[0], beam_width);

                for (token, log_prob) in top_k {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token);
                    all_candidates.push(BeamSequence::new(
                        new_tokens,
                        beam.score + log_prob,
                        beam.kv_blocks.as_ref().clone(),
                    ));
                }
            }

            if all_candidates.is_empty() {
                break;
            }

            // invariant: beam tokens are bounded by max_tokens; f32 precision
            // loss is acceptable for length-penalty normalization.
            all_candidates.sort_by(|a, b| {
                #[allow(clippy::cast_precision_loss)]
                let (sa, sb) = (
                    a.score / (a.tokens.len() as f32).powf(length_penalty),
                    b.score / (b.tokens.len() as f32).powf(length_penalty),
                );
                sb.partial_cmp(&sa)
                    .unwrap_or_else(|| sa.is_nan().cmp(&sb.is_nan()))
            });

            beams = all_candidates.into_iter().take(beam_width).collect();
        }

        let best = beams
            .into_iter()
            .max_by(|a, b| {
                #[allow(clippy::cast_precision_loss)]
                let (sa, sb) = (
                    a.score / (a.tokens.len() as f32).powf(length_penalty),
                    b.score / (b.tokens.len() as f32).powf(length_penalty),
                );
                sa.partial_cmp(&sb)
                    .unwrap_or_else(|| sa.is_nan().cmp(&sb.is_nan()))
            })
            .ok_or(crate::error::EngineError::EmptyBeamList)?;
        Ok(best)
    }

    fn get_top_k(logits: &[f32], k: usize) -> Vec<(TokenId, f32)> {
        let top_k_limit = k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or_else(|| a.1.is_nan().cmp(&b.1.is_nan()))
        });
        indexed
            .into_iter()
            .take(top_k_limit)
            .map(|(i, v)| (TokenId::try_from(i).unwrap_or(0), v))
            .collect()
    }
}
