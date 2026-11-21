//! Top-level speculative dispatch.
//!
//! [`Engine::step_speculative_inner`](super::super::Engine::step_speculative_inner)
//! is the speculative counterpart to `Engine::step`. It:
//!
//! 1. Builds the batch from the scheduler.
//! 2. Warms up draft KV caches after prefill.
//! 3. Generates draft tokens (per-seq via resolver, or batched legacy).
//! 4. Verifies drafts against target logits, accepting/rejecting.
//! 5. Rolls back KV cache for rejected drafts.
//! 6. Updates the scheduler with the produced tokens.
//! 7. Records speculative metrics (efficiency, accuracy, per-request rates).

use crate::error::Result;
use vllm_traits::{Batch, BatchPhase, FinishReason, SampledToken, SeqId};

impl crate::engine::Engine {
    /// Speculative decode step (called from `Engine::step` when speculative mode is on).
    ///
    /// **P36 v0.3 wire-type follow-up engine wire-through:** returns
    /// `Vec<(SeqId, SampledToken)>`. Speculative-accepted draft
    /// tokens carry a placeholder `SampledToken` with
    /// `logprob = NEG_INFINITY` + `top_logprobs = vec![]` (the
    /// logprob of an accepted draft would require re-running the
    /// target forward pass at each accepted position, which is
    /// non-trivial; computed-logprob for speculative accepted tokens
    /// is deferred to a future iteration). The bonus token (sampled
    /// by the verifier) carries the full `SampledToken` from
    /// `sample_one_with_params`. The HTTP handler detects the
    /// placeholder via a non-finite logprob
    /// (`!logprob.is_finite()`) and suppresses
    /// `ChatChoice::logprobs` output for any sequence containing one.
    pub(crate) fn step_speculative_inner(
        &mut self,
        max_draft: usize,
    ) -> Result<Vec<(SeqId, SampledToken)>> {
        let start = std::time::Instant::now();
        let mut batch = self.scheduler.build_batch();
        if batch.is_empty() {
            return Ok(vec![]);
        }

        self.ensure_verification_blocks(&mut batch, max_draft);

        // Warmup draft KV cache after prefill (Plans 17.4-A, 17.4-E)
        if batch.phase == BatchPhase::Prefill
            && self.speculative_mode
            && let Err(e) = self.warmup_draft_kv(&batch)
        {
            tracing::warn!(error = %e, "Draft warmup failed, continuing without warmup");
        }

        let draft_outputs = if self.draft_resolver.is_some() {
            // v18.0 per-request dispatch: resolve each seq's draft via the
            // resolver, then run draft generation per-seq. Mixed-routing
            // (RTE-03) and FALL-02 (runtime errors) live here.
            // `generate_per_seq_drafts` always returns Ok — per-seq errors
            // are caught internally and degrade the affected sequence. If
            // a future batch-wide failure mode is added, restore the Err
            // arm here to fall back to non-speculative decode.
            self.generate_per_seq_drafts(&batch, max_draft)
        } else {
            self.generate_batched_drafts(&batch, max_draft)?
        };

        let (verified, accepted_counts) =
            self.verify_draft_tokens_logits(&batch, &draft_outputs)?;

        // rejected-draft KV blocks are NOT rolled back here.
        // The verification span's blocks are pre-allocated
        // (`ensure_verification_blocks`) and the rejected
        // drafts' KV simply sits beyond `num_computed_tokens` — it is never
        // read and gets overwritten as the sequence grows. The old
        // `memory_rollback(rejected)` ran its block math against the
        // pre-step `num_computed_tokens`: for a resumed prefill (partial
        // prefix-cache hit) it rewound the count and released the matched
        // prefix block (the freed region is never recomputed → garbage KV),
        // and for decode it could free a block holding real prefix KV when
        // the rewind crossed a block boundary.

        // H-16 (PERF-05): pre-size `results` to the verified-sequence
        // count so the per-iteration push below does not reallocate.
        // a mid-chunk prefill's predicted token(s) are stale —
        // the real next prompt token is re-fed on the next chunk, so nothing
        // produced while the sequence is still mid-prompt may reach the
        // client stream. `verified` is a flat token list (a sequence can
        // contribute several entries: accepted drafts + bonus/rejection), so
        // staleness is keyed by seq_id via its batch entry (start + chunk <
        // prompt_len), read while the sequence is still in `running`
        // (update_speculative requeues partial prefills afterwards). The
        // final prefill chunk (transitions to Decoding) and decode sequences
        // are NOT masked.
        let stale_by_seq: std::collections::HashMap<SeqId, bool> = batch
            .seq_ids
            .iter()
            .enumerate()
            .map(|(i, sid)| {
                let stale = batch.is_prefill[i]
                    && self.scheduler.get_sequence(*sid).is_some_and(|s| {
                        batch.num_computed_tokens[i] + batch.input_tokens[i].len() < s.prompt_len
                    });
                (*sid, stale)
            })
            .collect();

        // Multi-token scheduler input tracking (Plan 17.1-E): fold the
        // emitted tokens into the scheduler per sequence, advancing
        // `num_computed_tokens` by the tokens whose KV was computed.
        //
        // The fold runs BEFORE the send loop so `seq.tokens` already holds
        // this step's tokens when they are about to be emitted — that is what
        // lets the send path recognise a completed `stop` sequence and
        // withhold it (OpenAI: "the returned text will not contain the stop
        // sequence"). The fold touches only scheduler state, so ordering it
        // ahead of the channel writes changes nothing it observes.
        let folded =
            self.fold_speculative_update(&batch, &verified, &accepted_counts, &stale_by_seq);

        // Emit what the fold KEPT, in burst order. A `stop` that completed
        // inside the burst trimmed its tail there, so the full burst must not
        // reach the send loop: without the trim it streams the stop sequence,
        // and re-creating a `pending_stream` entry for an already-ended
        // sequence would leave it there forever.
        let collected: Vec<(SeqId, SampledToken)> =
            keep_folded_tokens(&verified, &folded, &stale_by_seq);

        let mut disconnected = Vec::new();
        let results = self.emit_verified_tokens(&collected, &stale_by_seq, &mut disconnected);

        // End the sequences whose `stop` completed INSIDE their burst, before
        // the closed-channel pass (so a disconnect of this same step does not
        // finalize them twice under different reasons).
        self.finalize_mid_burst_stops(&folded);

        // cancel sequences whose response channel closed
        // (client disconnect) — after the fold, so the fold's per-sequence
        // frontier updates already ran; they must not keep burning the rest
        // of their `max_tokens` budget into a dead channel.
        self.cancel_on_closed_channels(&disconnected);

        // P38 v0.3 wire-type engine wire-through: stop-sequence
        // finalization. Must run after `scheduler.update` (so
        // `seq.tokens` includes the new tokens) and after the token-send
        // loop above (so matched tokens reach the client).
        self.finalize_stop_sequences(&batch);

        // Track accuracy in adaptive decoder and record adjustment events
        let total_draft: usize = draft_outputs.iter().map(std::vec::Vec::len).sum();
        let total_accepted: usize = accepted_counts.iter().sum();
        if let Some(ref mut decoder) = self.adaptive_decoder {
            if decoder.record_verification(total_draft, total_accepted) {
                self.scheduler.metrics.record_speculative_adjustment();
            }
        }

        // Record the batch-aggregated accepted/drafted ratio (Plan 17.4-F /
        // MTRC-02). acceptance is accepted/drafted (draft
        // quality — a perfect draft model scores 1.0, a never-accepted one
        // 0.0); the pre-fix draft/(draft+accepted) scored perfect drafts
        // 0.5 — an inverted metric where worse drafts report *higher*
        // acceptance. The duplicated `speculative_efficiency` gauge was
        // removed: it was fed this same ratio under a second
        // name, so `speculative_acceptance_rate` is the single wire name.
        if total_draft > 0 {
            self.scheduler
                .metrics
                .record_speculative_acceptance(total_accepted, total_draft);
        }

        // Record per-request acceptance rate (Plan 17.4-F / MTRC-01)
        for (i, seq_id) in batch.seq_ids.iter().enumerate() {
            let seq_drafts = draft_outputs[i].len();
            let seq_accepted = accepted_counts[i];
            self.scheduler
                .metrics
                .record_per_request_acceptance(*seq_id, seq_accepted, seq_drafts);
        }

        let finished = self.scheduler.finished_sequences();
        for seq in &finished {
            // Tell the handler the sequence stopped, then drop the
            // matching token channel. Sequences finalized above via
            // `finalize_stop_sequences` already had their txs removed
            // (idempotent `remove` → no-op), so this second pass only
            // affects max_tokens completions.
            self.finalize_finished(seq.id, FinishReason::Length);
            self.scheduler.metrics.remove_per_request(seq.id);
        }
        self.scheduler.clear_finished();

        if !batch.seq_ids.is_empty() {
            // count generated (output) tokens — the emitted
            // verified results — not the input token sum. `tokens_total`
            // ("Total tokens generated") must agree with the regular and
            // CUDA-graph step paths, which both count emitted results.
            //
            // exclude stale mid-chunk prefill predictions.
            // `record_tokens` counts REAL generated output only; the
            // pre-fix `results.len()` counted every verified entry
            // including stale ones (a mid-chunk prefill's draft + bonus
            // tokens that are just the verifier's guess at the next
            // prompt token, never real output), inflating `tokens_total`
            // by one ghost token per mid-chunk prefill step. The regular
            // path (`send_and_collect_results`) and the CUDA-graph path
            // already `continue` on stale so their `results` carry only
            // real output — this filter restores parity here. `results`
            // itself must stay complete (the fold below consumes every
            // entry to advance frontiers), so we only filter the COUNT.
            let generated = results
                .iter()
                .filter(|(seq_id, _)| !stale_by_seq.get(seq_id).copied().unwrap_or(false))
                .count();
            self.scheduler
                .metrics
                .record_tokens(u64::try_from(generated).unwrap_or(0));
            // prefill/decode phase split so the throughput
            // gauges are live on the speculative path too.
            self.scheduler.metrics.record_batch_phase_tokens(&batch);
            self.scheduler
                .metrics
                .record_batch_size(batch.seq_ids.len());
        }

        // invariant: elapsed millis fits in f64 mantissa (< 2^52 ms ≈ 142 years).
        #[allow(clippy::cast_precision_loss)]
        let elapsed = start.elapsed().as_millis() as f64;
        if elapsed > 0.0 {
            self.scheduler.metrics.record_latency(elapsed);
        }

        Ok(results)
    }

    /// Send each verified token to its sequence's client channel, gating out
    /// stale mid-chunk prefill predictions. `verified` is a
    /// flat token list — a sequence can contribute several entries (accepted
    /// drafts + bonus/rejection) — and staleness is keyed by `seq_id` via its
    /// batch entry (start + chunk < `prompt_len`).
    ///
    /// `results` still carries every verified entry so the caller's
    /// per-sequence scheduler fold can advance the mid-chunk prefill's
    /// frontier — only client emission is gated here.
    /// Shares the regular path's stop hold-back (`scheduler/batch.rs`): a
    /// token that may still turn out to be part of a `stop` sequence is
    /// buffered rather than sent, and the buffer is dropped when the tail
    /// completes a match. The caller folds this step's tokens into
    /// `seq.tokens` *before* calling this, so the suffix is readable here.
    fn emit_verified_tokens(
        &mut self,
        verified: &[(SeqId, SampledToken)],
        stale_by_seq: &std::collections::HashMap<SeqId, bool>,
        disconnected: &mut Vec<SeqId>,
    ) -> Vec<(SeqId, SampledToken)> {
        let mut results = Vec::with_capacity(verified.len());
        for (seq_id, sampled) in verified {
            let is_stale = stale_by_seq.get(seq_id).copied().unwrap_or(false);
            if is_stale {
                tracing::debug!(
                    seq_id = %seq_id,
                    token = %sampled.token,
                    "Drawing stale mid-chunk prediction (not real output); not emitting"
                );
                results.push((*seq_id, sampled.clone()));
                continue;
            }

            let params = self
                .scheduler
                .get_sequence(*seq_id)
                .map(|s| s.sampling_params.clone());
            let hold = self.stop_hold_back(params.as_ref());

            // Shares the regular path's `try_send_token` contract
            // (scheduler/batch.rs): `Full` (consumer drains slower than
            // the speculative path bursts) logs a WARN +
            // counts `dropped_tokens_total`; `Closed` (receiver dropped =
            // client disconnect) records the sequence in
            // `disconnected` so the caller cancels it. Pre-fix both were
            // silent no-ops and an aborted request generated into a
            // closed channel for its whole `max_tokens` budget.
            // A stop that landed exactly on the `max_tokens` boundary is
            // recorded by `check_completion`, which has already removed the
            // sequence from `running` — so it cannot be queried here. Consult
            // the record: the tail must stay withheld rather than be
            // released as ordinary output.
            let ended_on_stop = self.scheduler.stop_ended.contains(seq_id);
            let pending = self.scheduler.pending_stream.entry(*seq_id).or_default();
            pending.push(sampled.clone());
            if ended_on_stop {
                tracing::debug!(
                    seq_id = %seq_id,
                    held = pending.len(),
                    "Sequence reached max_tokens on a stop boundary; withholding the tail"
                );
                pending.clear();
                results.push((*seq_id, sampled.clone()));
                continue;
            }
            let mut released = Vec::new();
            while pending.len() > hold {
                released.push(pending.remove(0));
            }
            for ready in released {
                if let Some(tx) = self.response_txs.get(seq_id) {
                    self.try_send_token(*seq_id, &ready, tx, disconnected);
                }
            }
            results.push((*seq_id, sampled.clone()));
        }
        results
    }

    /// Fold the step's emitted tokens into the scheduler **per sequence**
    /// (Plan 17.1-E). `results` carries one entry per emitted token
    /// (accepted drafts + bonus/rejection token), and the scheduler must
    /// record every one of them; `num_computed_tokens` must advance by the
    /// number of tokens whose KV the target model actually computed during
    /// verification (`input_len + accepted`).
    ///
    /// the old code flattened `results` into per-token
    /// `seq_ids`/`sampled` vectors but passed a per-sequence
    /// `input_counts` vector. `scheduler.update` zips the two together, so
    /// the fold truncated to one token per sequence — tokens after the first
    /// were streamed to the client but never recorded in `seq.tokens` — and
    /// `num_computed_tokens` advanced by `accepted+1` (only correct for
    /// decode batches with `input_len` == 1). Long prompts in speculative mode
    /// never completed prefill in one step and re-fed already-generated
    /// draft tokens back into subsequent prefill batches.
    fn fold_speculative_update(
        &mut self,
        batch: &Batch,
        results: &[(SeqId, SampledToken)],
        accepted_counts: &[usize],
        stale_by_seq: &std::collections::HashMap<SeqId, bool>,
    ) -> std::collections::HashMap<SeqId, FoldedBurst> {
        let mut per_seq: Vec<(SeqId, Vec<SampledToken>)> = Vec::new();
        let mut seq_index: std::collections::HashMap<SeqId, usize> =
            std::collections::HashMap::new();
        let mut folded: std::collections::HashMap<SeqId, FoldedBurst> =
            std::collections::HashMap::new();
        for (seq_id, sampled) in results {
            if let Some(&i) = seq_index.get(seq_id) {
                per_seq[i].1.push(sampled.clone());
            } else {
                seq_index.insert(*seq_id, per_seq.len());
                per_seq.push((*seq_id, vec![sampled.clone()]));
            }
        }

        // Fold every sequence in ONE `update_speculative` call. That call ends
        // by running `finalize_finished_sequences` + `requeue_partial_prefills`
        // over the WHOLE running set, so calling it per sequence moved the
        // not-yet-folded ones out from under the loop: a freshly admitted
        // prefill sits at `num_computed_tokens == 0` (below its prompt
        // length), so it looked like a mid-chunk prefill and was requeued
        // with this step's already-emitted tokens dropped from `seq.tokens`.
        // The next round then generated those tokens again, so a client saw
        // its output twice and the sequence overshot `max_tokens`.
        let mut fold_ids: Vec<SeqId> = Vec::with_capacity(per_seq.len());
        let mut fold_input_counts: Vec<usize> = Vec::with_capacity(per_seq.len());

        for (seq_id, tokens) in &per_seq {
            let Some(i) = batch.seq_ids.iter().position(|sid| sid == seq_id) else {
                continue;
            };
            let chunk_len = batch.input_tokens.get(i).map_or(1, std::vec::Vec::len);
            // a MID-chunk prefill's accepted drafts occupy real
            // upcoming prompt positions — the verifier only checked them
            // against the target model's continuation guess, never against
            // the prompt itself, so their KV is guessed content and the real
            // prompt tokens would be SKIPPED if the frontier advanced past
            // them (`requeue_partial_prefills`/ preserves the inflated
            // frontier, so the next chunk composes from it). Advance the
            // frontier by the REAL chunk only; the final chunk (completes the
            // prompt) and decode sequences keep the full
            // `chunk_len + accepted` — there the drafts are genuine generated
            // continuation starting at `prompt_len`. `stale_by_seq` encodes
            // exactly this mid-chunk predicate (start + chunk < prompt_len).
            let mid_chunk = stale_by_seq.get(seq_id).copied().unwrap_or(false);
            // The burst can also complete a `stop` sequence, and it is
            // folded into `seq.tokens` as ONE update: by the time the send
            // loop and `finalize_stop_sequences` look at the sequence the
            // match is no longer at the tail, so neither of them can see it.
            // Cut the burst here, at the fold point — the matching stop and
            // everything after it are output the client must never receive.
            // A mid-chunk prefill's entries are stale predictions the send
            // loop discards, so a stop "match" in them is not output and
            // must not end the sequence.
            let trim = if mid_chunk {
                BurstTrim {
                    kept: tokens.clone(),
                    match_starts_before_burst: false,
                }
            } else {
                self.trim_burst_at_stop(*seq_id, tokens)
            };
            let match_starts_before_burst = trim.match_starts_before_burst;
            let kept = trim.kept;
            let stopped = kept.len() < tokens.len();
            // `input_count` is how many positions the TARGET model actually
            // computed KV for — the `input_len + drafts` verification span —
            // NOT how many tokens were kept for the client. A stop that cut
            // the burst still had its trailing positions computed, and the
            // sequence is finalized immediately afterwards
            // (`finalize_mid_burst_stops`), so the frontier it leaves behind
            // is never read again. Capping this by `kept` would state the
            // wrong thing about the KV cache for no behavioural gain.
            let input_count = if mid_chunk {
                chunk_len
            } else {
                chunk_len + accepted_counts[i]
            };
            fold_ids.push(*seq_id);
            fold_input_counts.push(input_count);
            folded.insert(
                *seq_id,
                FoldedBurst {
                    tokens: kept,
                    stopped,
                    match_starts_before_burst,
                },
            );
        }
        // `folded` also supplies the per-sequence token groups the scheduler
        // needs, so nothing is cloned twice.
        let fold_tokens: Vec<Vec<SampledToken>> = fold_ids
            .iter()
            .filter_map(|seq_id| folded.get(seq_id).map(|burst| burst.tokens.clone()))
            .collect();
        if !fold_ids.is_empty() {
            self.scheduler
                .update_speculative(&fold_ids, &fold_tokens, &fold_input_counts);
        }
        folded
    }

    /// End every sequence whose `stop` completed INSIDE its speculative
    /// burst, reporting [`FinishReason::Stop`].
    ///
    /// This replaces [`Self::finalize_stop_sequences`] for those sequences:
    /// that pass suffix-matches the folded sequence and therefore cannot see
    /// a stop the fold already cut away. It runs after the send loop, so the
    /// tokens generated before the stop reach the client first.
    ///
    /// `stop_ended` is recorded only when the match began in an earlier
    /// burst. The send loop holds back the trailing `stop_len - 1` tokens of
    /// every burst precisely because they may start a stop: when one of them
    /// did, they ARE the stop and `finalize_finished` must discard them. A
    /// match contained entirely in this burst leaves only ordinary output in
    /// that buffer, which must be flushed instead.
    fn finalize_mid_burst_stops(&mut self, folded: &std::collections::HashMap<SeqId, FoldedBurst>) {
        for (seq_id, burst) in folded {
            if !burst.stopped {
                continue;
            }
            tracing::debug!(
                seq_id = %seq_id,
                kept = burst.tokens.len(),
                "stop completed inside a speculative burst; ending the sequence"
            );
            if burst.match_starts_before_burst {
                self.scheduler.stop_ended.insert(*seq_id);
            }
            self.scheduler.finish_sequence(*seq_id);
            self.finalize_finished(*seq_id, FinishReason::Stop);
        }
    }

    /// Cut a speculative burst at the first point where it reaches a
    /// terminator: a configured `stop` token sequence, or the model's own
    /// EOS token.
    ///
    /// The burst is returned untouched when the sequence is unknown, when it
    /// has no stops and the engine has no EOS id, or when nothing matches.
    /// On a match everything after the terminator is dropped — those tokens
    /// were produced by a sequence that had already ended — and:
    ///
    /// - a `stop` sequence is dropped together with its trailing tokens
    ///   (`OpenAI`: "the returned text will not contain the stop sequence");
    /// - the EOS token itself is KEPT, matching `step_regular`, where it
    ///   reaches the response channel and the HTTP layer drops its text
    ///   (`should_skip_token_text`).
    ///
    /// `match_starts_before_burst` reports whether the match began in tokens
    /// generated by an EARLIER step, which a multi-token stop can do.
    ///
    /// The window is anchored at `seq.prompt_len`, mirroring
    /// [`Self::finalize_stop_sequences`]: a stop-shaped suffix of the PROMPT
    /// is never a match.
    fn trim_burst_at_stop(&self, seq_id: SeqId, tokens: &[SampledToken]) -> BurstTrim {
        let unchanged = BurstTrim {
            kept: tokens.to_vec(),
            match_starts_before_burst: false,
        };
        let Some(seq) = self.scheduler.get_sequence(seq_id) else {
            return unchanged;
        };
        let stops: &[Vec<vllm_traits::TokenId>] = seq
            .sampling_params
            .stop_token_sequences
            .as_deref()
            .unwrap_or(&[]);
        if stops.is_empty() && self.eos_token_id.is_none() {
            return unchanged;
        }
        // Only the newly generated region can match; the prompt's own tail is
        // excluded exactly as in `finalize_stop_sequences`.
        let mut window: Vec<vllm_traits::TokenId> =
            seq.tokens.get(seq.prompt_len..).unwrap_or(&[]).to_vec();
        for (idx, sampled) in tokens.iter().enumerate() {
            window.push(sampled.token);
            let matched = idx + 1;
            if let Some(stop_len) = crate::sampling::matched_stop_len(&window, stops) {
                // The match may reach back into tokens generated by an
                // earlier step, in which case this burst contributes only the
                // stop's last `matched` tokens. Cut from where it began.
                let keep = matched.saturating_sub(stop_len);
                tracing::debug!(
                    seq_id = %seq_id,
                    kept = keep,
                    stop_len,
                    match_starts_before_burst = stop_len > matched,
                    "stop completed inside a speculative burst; trimming the tail"
                );
                return BurstTrim {
                    kept: tokens[..keep].to_vec(),
                    match_starts_before_burst: stop_len > matched,
                };
            }
            if self.eos_token_id == Some(sampled.token) {
                tracing::debug!(
                    seq_id = %seq_id,
                    kept = matched,
                    "EOS generated inside a speculative burst; trimming the tail"
                );
                return BurstTrim {
                    kept: tokens[..matched].to_vec(),
                    match_starts_before_burst: false,
                };
            }
        }
        unchanged
    }

    /// Grow each batch sequence's block table to cover the full speculative
    /// verification span (`num_computed + input_len + max_draft` tokens)
    /// BEFORE the draft/target forward writes KV.
    ///
    /// The batch composer pre-allocates blocks for `input_len` only; the
    /// verifier then processes `input_len + drafts` tokens, and
    /// `write_prefill_kv`'s missing-block fallback
    /// (`block_ids.get(block_idx).unwrap_or(0)`) silently wrote the overflow
    /// draft KV into block 0 — corrupting the prompt's first-block KV
    /// whenever the draft span crossed a block boundary (verified by a
    /// real-model regression test). The extra blocks stay owned by the
    /// sequence (released on finish); rejected-draft blocks are simply
    /// unused capacity beyond `num_computed_tokens` (— the old
    /// post-verification `memory_rollback` is no longer needed and was
    /// actively harmful for resumed prefill / boundary-crossing decode).
    fn ensure_verification_blocks(&mut self, batch: &mut vllm_traits::Batch, max_draft: usize) {
        for (i, seq_id) in batch.seq_ids.iter().copied().enumerate() {
            let base = batch.num_computed_tokens.get(i).copied().unwrap_or(0);
            let span = batch.input_tokens.get(i).map_or(0, std::vec::Vec::len) + max_draft;
            let blocks_needed = (base + span).div_ceil(vllm_traits::BLOCK_SIZE);
            if batch.kv_block_ids[i].len() < blocks_needed {
                self.scheduler
                    .ensure_blocks_for_tokens(seq_id, blocks_needed);
                if let Some(seq) = self.scheduler.get_sequence(seq_id) {
                    batch.kv_block_ids[i].clone_from(seq.kv_blocks.as_ref());
                }
            }
        }
    }
}

/// The tokens the fold kept for the client, in burst order.
///
/// `verified` is flat (a sequence contributes several entries per burst) and
/// `folded` holds the post-trim list per sequence, so the two are walked in
/// lockstep: every burst entry for a sequence takes the next kept token, and
/// entries past a trim simply run out. Stale mid-chunk prefill predictions
/// are dropped, as in the send loop.
fn keep_folded_tokens(
    verified: &[(SeqId, SampledToken)],
    folded: &std::collections::HashMap<SeqId, FoldedBurst>,
    stale_by_seq: &std::collections::HashMap<SeqId, bool>,
) -> Vec<(SeqId, SampledToken)> {
    let mut cursor: std::collections::HashMap<SeqId, usize> = std::collections::HashMap::new();
    verified
        .iter()
        .filter(|(seq_id, _)| !stale_by_seq.get(seq_id).copied().unwrap_or(false))
        .filter_map(|(seq_id, _)| {
            let next = cursor.entry(*seq_id).or_insert(0);
            let token = folded.get(seq_id)?.tokens.get(*next)?.clone();
            *next += 1;
            Some((*seq_id, token))
        })
        .collect()
}

/// Result of cutting a speculative burst at a `stop` match.
struct BurstTrim {
    /// Burst tokens preceding the match — the only ones the client may
    /// receive.
    kept: Vec<SampledToken>,
    /// The match began in tokens generated by an earlier step, so its opening
    /// tokens sit in the send loop's hold-back buffer rather than in this
    /// burst.
    match_starts_before_burst: bool,
}

/// What a speculative burst contributed to one sequence after the
/// terminator trim: the tokens kept for the client, and whether the fold cut
/// the burst short (a `stop` or the EOS token completed inside it).
struct FoldedBurst {
    tokens: Vec<SampledToken>,
    stopped: bool,
    match_starts_before_burst: bool,
}
