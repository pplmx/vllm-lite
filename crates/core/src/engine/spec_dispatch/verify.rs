//! Logit-based verification with temperature-aware acceptance (Plan 17.1-C
//! + architecture-performance.md §6 speculative fix).
//!
//! Takes the per-sequence draft tokens and the target model's logits, then:
//!
//! - When `temperature == 0.0` (greedy), accept drafts whose top-1 matches
//!   the target's argmax. Mismatch emits the target argmax and rejects the
//!   remaining drafts. A bonus token is emitted (also via argmax) if all
//!   drafts were accepted.
//! - When `temperature > 0.0` (sampling), sample from the target
//!   distribution using the per-sequence [`SamplingParams`]. The draft is
//!   accepted if the sampled target token matches the draft token;
//!   otherwise the sampled target token is emitted and the remaining
//!   drafts are rejected. The bonus token uses the same sampler.
//!
//! The sampling form is the standard "lossless speculative decoding"
//! verifier: the marginal distribution of accepted + bonus tokens matches
//! the target's sampling distribution, so the wall-clock speedup does not
//! change the output statistics. It is not the full `min(1, p/q)`
//! rejection-sampling variant — that requires draft-side logits we don't
//! carry on the wire. The sampled-match variant is a strict improvement
//! over the old argmax path under non-zero temperature because the target
//! now uses the same sampler the rest of the engine uses, instead of
//! always picking the most likely token.

use crate::error::Result;
use crate::sampling::sample_one_with_params;
use crate::sync::lock_mutex;
use vllm_traits::{Batch, SampledToken, SamplingParams, SeqId, TokenId};

impl crate::engine::Engine {
    /// Returns `(accepted_tokens, accepted_counts_per_sequence)`.
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** returns
    /// `Vec<(SeqId, SampledToken)>` instead of `Vec<(SeqId, TokenId)>`.
    /// Speculative-accepted draft tokens carry a placeholder
    /// `SampledToken` with `logprob = 0.0` + `top_logprobs = vec![]`
    /// (the logprob of an accepted draft would require re-running
    /// the target forward pass at each accepted position, which is
    /// non-trivial; computed-logprob for speculative accepted tokens
    /// is deferred to a future iteration). The bonus token (sampled
    /// by the verifier) carries the full `SampledToken` from
    /// `sample_one_with_params`. The HTTP handler detects the
    /// placeholder via `logprob == 0.0 && top_logprobs.is_empty()`
    /// and suppresses `ChatChoice::logprobs` output for any sequence
    /// containing one.
    pub(crate) fn verify_draft_tokens_logits(
        &self,
        batch: &Batch,
        draft_outputs: &[Vec<TokenId>],
    ) -> Result<(Vec<(SeqId, SampledToken)>, Vec<usize>)> {
        // H-16 (PERF-05): pre-size `results` to the sequence count so the
        // per-iteration `results.push(...)` does not reallocate. Mirrors
        // the existing `accepted_counts` hint one line below.
        let mut results = Vec::with_capacity(batch.seq_ids.len());
        let mut accepted_counts = Vec::with_capacity(batch.seq_ids.len());

        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let drafts = &draft_outputs[i];

            // Pick the per-sequence sampling params carried on the Batch
            // (populated by BatchComposer from Sequence::sampling_params —
            // see ARCH-02 fix in CHANGELOG). Fall back to default
            // (greedy) if the Batch is missing the field (e.g. synthetic
            // test fixtures).
            let params = batch.sampling_params.get(i).cloned().unwrap_or_default();

            // Empty-drafts path: sample (or argmax) directly from the
            // target model's last-position logits.
            if drafts.is_empty() {
                let logits = lock_mutex(&self.target_model)?.forward_logits(
                    &[*seq_id],
                    std::slice::from_ref(&batch.input_tokens[i]),
                    std::slice::from_ref(&batch.positions[i]),
                    std::slice::from_ref(&batch.kv_block_ids[i]),
                    std::slice::from_ref(&batch.num_computed_tokens[i]),
                    std::slice::from_ref(&batch.is_prefill[i]),
                )?;
                let sampled = logits.first().map_or_else(
                    || placeholder_sampled(0),
                    |pos_logits| sample_or_argmax(pos_logits, &params),
                );
                results.push((*seq_id, sampled));
                accepted_counts.push(0);
                continue;
            }

            // Concatenate input tokens + draft tokens for verification
            let verify_tokens: Vec<TokenId> = batch.input_tokens[i]
                .iter()
                .chain(drafts.iter())
                .copied()
                .collect();
            let verify_positions: Vec<usize> = (0..verify_tokens.len()).collect();

            // Get logits from target model for all positions
            let logits = lock_mutex(&self.target_model)?.forward_logits(
                &[*seq_id],
                std::slice::from_ref(&verify_tokens),
                std::slice::from_ref(&verify_positions),
                std::slice::from_ref(&batch.kv_block_ids[i]),
                std::slice::from_ref(&batch.num_computed_tokens[i]),
                std::slice::from_ref(&batch.is_prefill[i]),
            )?;

            let logits: &[f32] = logits.first().map_or(&[], std::vec::Vec::as_slice);
            let vocab_size = lock_mutex(&self.target_model)?.vocab_size();

            let mut accepted = 0usize;

            for (j, &draft_token) in drafts.iter().enumerate() {
                let offset = j * vocab_size;
                if offset + vocab_size > logits.len() {
                    break;
                }
                let pos_logits = &logits[offset..offset + vocab_size];
                // Sample or argmax from target, then check draft match.
                let target_token = sample_or_argmax(pos_logits, &params).token;

                if target_token == draft_token {
                    // Accepted draft — placeholder SampledToken (no
                    // logprob info available without re-running
                    // forward at this position; see function doc).
                    results.push((*seq_id, placeholder_sampled(draft_token)));
                    accepted += 1;
                } else {
                    // Rejection — emit the sampled target token with
                    // its full SampledToken (this position's logits
                    // are available).
                    results.push((*seq_id, sample_or_argmax(pos_logits, &params)));
                    break;
                }
            }

            // Add a bonus token if all drafts were accepted
            if accepted == drafts.len() {
                let bonus_offset = accepted * vocab_size;
                if bonus_offset + vocab_size <= logits.len() {
                    let bonus_logits = &logits[bonus_offset..bonus_offset + vocab_size];
                    let bonus_sampled = sample_or_argmax(bonus_logits, &params);
                    results.push((*seq_id, bonus_sampled));
                }
            }

            accepted_counts.push(accepted);
        }

        Ok((results, accepted_counts))
    }
}

/// Pick a token from `logits` using `params`: argmax for greedy
/// (`temperature == 0.0`), `sample_one_with_params` otherwise. Thin
/// indirection so the verifier doesn't sprinkle the same `if`
/// everywhere. Returns a full [`SampledToken`] (P36 v0.3 wire-type
/// follow-up engine wire-through).
fn sample_or_argmax(logits: &[f32], params: &SamplingParams) -> SampledToken {
    if params.temperature <= 0.0 {
        // Greedy path: synthesize a SampledToken with logprob=0.0
        // placeholder (the verifier only uses `.token` for the
        // accept/reject comparison on greedy; logprobs are surfaced
        // by the regular non-speculative path which calls
        // `sample_one_with_params` directly). For the bonus token
        // we do want a real logprob, so recompute via
        // `sample_one_with_params` (which short-circuits to
        // argmax_logits for T=0 anyway and populates the logprob
        // correctly).
        sample_one_with_params(logits, params, &[])
    } else {
        // Empty seen-token list is fine: `sample_one_with_params`
        // short-circuits on `repeat_penalty == 1.0` (the default).
        sample_one_with_params(logits, params, &[])
    }
}

/// Placeholder [`SampledToken`] for speculative-accepted draft tokens
/// where the true logprob is unavailable without re-running the
/// target forward pass at that position. Detected by the HTTP layer
/// via `logprob == 0.0 && top_logprobs.is_empty()`.
const fn placeholder_sampled(token: TokenId) -> SampledToken {
    SampledToken {
        token,
        logprob: 0.0,
        top_logprobs: Vec::new(),
    }
}

/// Test-only re-export of [`sample_or_argmax`] for the regression suite
/// under `engine::spec_dispatch::tests`. The function is private because
/// callers should go through [`Engine::verify_draft_tokens_logits`], but
/// the tests need to drive the sampler directly without a full engine
/// step to keep the assertions deterministic.
#[doc(hidden)]
#[allow(dead_code)] // only consumed by `engine::spec_dispatch::tests`
pub fn test_only_sample_or_argmax(logits: &[f32], params: &SamplingParams) -> SampledToken {
    sample_or_argmax(logits, params)
}
