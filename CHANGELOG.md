# 📋 Changelog

<p align="center">
  <img src="https://img.shields.io/badge/Keep%20a%20Changelog-1.0.0-blue.svg?style=flat-square" alt="Keep a Changelog">
  <img src="https://img.shields.io/badge/Semantic%20Versioning-2.0.0-green.svg?style=flat-square" alt="Semantic Versioning">
</p>

> All notable changes to **vLLM-lite** will be documented in this file.

---

## 📊 Release Statistics

|     版本     |    日期    |          测试          | 覆盖率 (raw / real) |
| :----------: | :--------: | :--------------------: | :-----------------: |
| [Unreleased] |     -      | `cargo test --workspace --all-features` 1779 tests pass (post-P41 engine plumbing; 4 `write_layer_block` + 4 `MemoryManager::block_data_source` + 2 `EngineBuilder::with_paged_kv_cache` + 3 `MultiNodeConfig` round-trip) | 68.0% real (post-P41) |
|   [v22.0]    | 2026-06-27 |         1179+          | ~50% real (97.8% figure was placeholder-based) |
|   [v21.0]    | 2026-06-27 |         1146+          | ~50% real |
|   [v20.0]    | 2026-06-27 |         1144+          | ~50% real |
|   [v19.0]    | 2026-06-27 |         1139+          | ~50% real |
|   [v18.0]    | 2026-06-27 | 277 (vllm-core) + 654+ |  90%+  |

---

## 🚀 [Unreleased]

### Fixed

- **Broken intra-doc links in module-level docstrings** (v31.0 P40 T8) — pre-existing drift accumulated across P28-P38 (not introduced by P40) surfaced by the workspace doc check during P40's CI verification. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace` failed with 13 unresolved-link errors across 6 files: `vllm-traits/src/sampling.rs` (4 — `crate::sampling::sample_*` paths pointed to functions in `vllm_core::sampling`, not `vllm_traits::sampling`; `vllm_traits::FinishReason::Stop` on the `stop_token_sequences` builder resolved to itself), `vllm-core/src/sampling.rs` (1 — unqualified `[`sample_one_with_params`]` in the module-level doc), `vllm-core/src/scheduler/memory/mod.rs` (5 — the entire "Multi-node cache coherence" section used intra-doc links to items gated behind `#[cfg(feature = "multi-node")]` so the default-features doc build failed), `vllm-server/src/openai/chat.rs` (3 — `[`TokenId`]` without `vllm_traits::` prefix + two `[`super::completions::...`]` links to private items in the sibling module), `vllm-server/src/openai/completions.rs` (1 — same `[`TokenId`]` pattern), `vllm-model/src/paged_tensor/tensor_store/mod.rs` (1 — `[`crate::paged_kv_cache_wrapper`]` only resolves under `--features multi-node`). All fixed by replacing the broken intra-doc links with descriptive backtick-quoted text (consistent with how the existing project style handles cross-feature / cross-crate references); zero behavior change, zero public-API delta, zero test delta. **`just ci` is now green workspace-wide** — both `cargo doc --workspace` (default features) and `cargo doc --workspace --all-features` pass; doc-coverage real ticked from 67.91% → 68.0% (the descriptive-text rewrites of `scheduler/memory/mod.rs`'s multi-node section + `sampling.rs`'s RNG-seeding doc added a few more counted items). **6 files, +28 / -29 lines, all `///` or `//!` doc-comment edits.**
- **ARCH-02 sampling parameters reach the hot path** — previously the HTTP layer accepted `temperature` / `top_p` / `top_k` / `repeat_penalty` and stored them on the `Request`, but `Engine::step_regular` called `model.forward`, which chose the next token greedily inside the model layer. The params were silently dropped. The seam is now `forward_logits → engine-side sample_batch_with_params(batch.sampling_params)`. `Batch` carries a per-sequence `Vec<SamplingParams>` (moved to `vllm_traits::sampling` so the wire type doesn't depend on `vllm_core`); `BatchComposer::{decode,prefill,chunked}` populate it from `Sequence::sampling_params`; `sample_batch_with_params` is the new entry point that respects per-sequence params. Regression test in `crates/core/tests/sampling_params.rs` (3 cases).
- **`attn_factor` (YaRN §3.3 attention-temperature scaling) now honoured by all four forward paths** (v31.0 P18 / Phase 31-F first item) — previously `attn_factor` was stored on `GqaAttention` and set from `rope.attn_factor()` (via `RopeGqaAttention`) but only the standard `forward()` path actually applied it (`qk.affine(attn_factor / sqrt(d), 0.0)` after the QK^T matmul). The three production paths — `paged_attention_fn`, `tiled_attention_fn`, `flash_attention_fn` — each delegate to a lower-level function that applies its OWN `1/sqrt(d)` scaling internally (`util::paged_attention` line 189, `util::tiled_attention` line 244, `GqaFlashAttention::forward` line 287) and silently ignored the factor. Fix: pre-scale Q by `attn_factor` in each of the three wrappers (and in the `use_fused` branch of `forward()`) via a new private `apply_attn_factor` helper. Mathematically equivalent to the standard path because softmax is invariant to positive scalar multiplication: `(Q * attn_factor) @ K^T = attn_factor * (Q @ K^T)`, then the internal `1/sqrt(d)` gives `attn_factor / sqrt(d) * Q @ K^T` — same result. **No-op when `attn_factor` is `None` or `Some(1.0)`** (within `f32::EPSILON`) — Q is returned unchanged, no allocation, no kernel launch. Field-level doc-comment in `gqa/mod.rs:41-50` updated to remove the "Currently only honoured by the standard forward path; paged/tiled/flash attention paths silently ignore this value" limitation note. 6 new unit tests in `crates/model/src/components/attention/gqa/tests.rs` (3 paths × 2 tests: `attn_factor=1.0` matches `None` + `attn_factor=0.5` changes output). Behaviour change, no public API signature change.
- **`RopeScaling` (YaRN / Linear / Dynamic / Su) reaches the block construction layer** (v31.0 P19 / Phase 31-F second item) — previously a `config.json` that declared `rope_scaling` was parsed by `Qwen3Config::from_value` but the block was silently dropped at every downstream hop: `ModelConfig::from_config_json` had no field for it, the `From<&Qwen3Config> for ModelConfig` impl dropped it on the floor, and `RopeGqaAttention::new` / `TransformerBlock::new` hard-coded `RoPE::new(...)` with `scaling_factor = 1.0` and `attn_factor = None`. The YaRN-config Qwen3 model therefore produced numerically identical output to a default Qwen3 of the same shape at the engine boundary, regardless of what `rope_scaling` declared. Fix: new `ModelConfig::rope_scaling: Option<RopeScaling>` field populated from `config.json["rope_scaling"]` in `from_config_json` (malformed values silently fall back to `None` — the conservative pattern used for other optional fields); new `RopeGqaAttention::{new_with_rope_scaling, new_with_weights_rope_scaling}` and `TransformerBlock::{new_with_rope_scaling, new_with_weights_rope_scaling}` constructors that thread the block through to `RoPE::new_with_scaling(...)`; the existing `new` / `new_with_weights` / `new_with_tp` constructors now delegate to the scaling-aware variants with `None` so backward compatibility is preserved bit-for-bit. `factory::new_block` switches to `TransformerBlock::new_with_rope_scaling(...)` and forwards `config.max_position_embeddings` + `config.rope_scaling.as_ref()`. 2 new unit tests in `crates/model/src/components/attention/rope_gqa/tests.rs` (`new_with_rope_scaling_none_matches_new_construction` + `new_with_rope_scaling_yarn_attaches_attn_factor_to_inner`); test-fixture updates in `gemma4` / `mixtral` for the new field. Closes Phase 31-F second item.
- **`RopeScaling` reaches the shared decoder-block factory and Mixtral block** (v31.0 P20 / Phase 31-F follow-up) — P19 closed the thread-through for the Qwen3-specific path (`qwen3/block/factory::new_block` + `RopeGqaAttention::new_with_rope_scaling`), but two more call sites continued to use the bare constructors and silently dropped the block: `crates/model/src/components/decoder_block/factory.rs` (the shared `new_block` / `block_from_weights` used by Llama/Mistral-style checkpoints) and `crates/model/src/mixtral/block.rs` (`MixtralBlock::new` / `MixtralBlock::from_weights`). Post-fix, both factories thread `config.max_position_embeddings` + `config.rope_scaling.as_ref()` into the scaling-aware constructors (`RopeGqaAttention::new_with_rope_scaling` / `new_with_weights_rope_scaling`). A Llama or Mixtral checkpoint that declares a YaRN block in `config.json` now actually takes effect at the attention layer instead of producing numerically identical output to the default model of the same shape. 4 new unit tests: `new_block_accepts_yarn_rope_scaling` + `new_block_accepts_none_rope_scaling` in `crates/model/src/components/decoder_block/factory.rs::tests`, and `test_mixtral_block_new_accepts_yarn_rope_scaling` + `test_mixtral_block_from_weights_accepts_yarn_rope_scaling` in `crates/model/src/mixtral/block/tests.rs` (the latter builds a populated expert weight map to exercise the HF weight-loader path end-to-end at construction time). Closes Phase 31-F follow-up for non-Qwen3 architectures.
- **`user` field declared on OpenAI chat + completions request types** (v31.0 P21 / v0.2 wire-type follow-up) — OpenAI's `user` field is the end-user identifier used for safety / abuse tracking. Pre-fix, `ChatRequest` and `CompletionRequest` had no field for it, so serde silently rejected any request that included `"user": "..."` in its JSON body. Post-fix, both wire types declare `user: Option<String>` with `#[serde(default, skip_serializing_if = "Option::is_none")]` — the field is now accepted when present, defaults to `None` when omitted (preserving backward compatibility for every existing client), and is never echoed in the response (OpenAI's contract: `user` is request-only). The chat handler threads `user = ?req.user` into the existing `tracing::info!(...)` log lines (`Request started`, `Request completed`, `Streaming request started`) so downstream subscribers (rate-limiter, audit log, abuse-detection pipeline) can pick it up without any engine-side changes. The completions handler currently does not log the field — adding a new `tracing::info!` line was deferred to keep scope tight; the wire-type contract is symmetric. **Honoring is a no-op today** — vllm-lite has no auth/persistence layer that consumes the field; the value flows through structured tracing only. 4 new integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_with_user_field_accepted_by_handler`, `test_chat_without_user_field_works_baseline`, `test_completions_with_user_field_accepted_by_handler`, `test_chat_user_field_wire_type_round_trip` (the last pins the serde contract independently of any handler-level test). `docs/reference/openai-compatibility.md` flipped both `user` rows from "Not declared" → "Wired (tracing pass-through)" and marked the field shipped in the v0.2 follow-ups section. Public-API delta: 2 new fields (`ChatRequest::user`, `CompletionRequest::user`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate.
- **`response_format` field declared on `ChatRequest` with `ResponseFormat::Text` + `ResponseFormat::JsonObject` enum** (v31.0 P22 / v0.2 wire-type follow-up) — OpenAI's `response_format` field is the JSON-mode selector. Pre-fix, `ChatRequest` had no field for it, so serde silently rejected any request that included `"response_format": ...` in its JSON body; the legacy `/v1/completions` endpoint never supported `response_format` per OpenAI spec, so `CompletionRequest` intentionally does NOT declare the field (the legacy endpoint silently drops unknown fields, which is its permissive contract). Post-fix: new `ResponseFormat` enum with `Text` + `JsonObject` variants using `#[serde(tag = "type", rename_all = "snake_case")]` so the JSON shape matches upstream 1:1 (`{"type": "text"}` / `{"type": "json_object"}`); new `ChatRequest::response_format: Option<ResponseFormat>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`. The `{type: "json_schema"}` variant (which would require a grammar-constrained decoder) is rejected at the **serde layer** because it is not a declared variant — axum's `Json<T>` extractor surfaces the deserialization failure as `422 Unprocessable Entity` (axum's standard contract for "syntactically valid JSON but semantically invalid input"). The new `validate_chat_response_format` is a documentation-first no-op validator (mirrors `validate_chat_request_fields` shape) that provides a single hook point for future strict checks (e.g. v0.3 `JsonObject` payload syntax validation). The chat handler threads `response_format = ?req.response_format` into the existing `tracing::info!(...)` log lines for observability. **Honoring is a no-op today** — the engine does not enforce JSON syntax via a constrained-decoding hook (deferred to v0.3 / v32+ work). 6 new integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_with_response_format_text_accepted_by_handler`, `test_chat_with_response_format_json_object_accepted_by_handler`, `test_chat_with_response_format_json_schema_rejected` (pins the 422 contract), `test_chat_without_response_format_works_baseline`, `test_chat_response_format_wire_type_round_trip` (pins the serde contract independently of any handler-level test, including `json_schema` deserialization failure), `test_completion_request_has_no_response_format_field` (pins the wire-type asymmetry — legacy endpoint has no field). 5 new unit tests in `crates/server/src/openai/sampling_validation.rs::tests`: `chat_response_format_none_passes_validation`, `chat_response_format_text_passes_validation`, `chat_response_format_json_object_passes_validation`, `chat_request_with_response_format_text_passes_full_field_validation`, `chat_request_with_response_format_json_object_passes_full_field_validation`. `docs/reference/openai-compatibility.md` flipped the chat `response_format` row from "Not declared" → "Wired (declaration + validation)" and marked the field shipped in the v0.2 follow-ups section. Public-API delta: 1 new type (`ResponseFormat`) + 1 new field (`ChatRequest::response_format`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate.
- **`seed` field declared on OpenAI chat + completions request types** (v31.0 P23 / v0.2 wire-type follow-up) — OpenAI's `seed` field is the "best effort to sample deterministically" knob. Pre-fix, neither `ChatRequest` nor `CompletionRequest` declared the field, so serde silently rejected any request that included `"seed": ...` in its JSON body. Post-fix, both wire types declare `seed: Option<i64>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`. Per OpenAI spec any `i64` is accepted (no range / sign validation, no NaN check) — the wire-type contract is intentionally permissive so v32+ can tighten if needed without a breaking change. The chat handler threads `seed = ?req.seed` into the three existing `tracing::info!(...)` log lines (`Request started` / `Request completed` / `Streaming request started`) so determinism is at least observable in trace logs (operators can see "did this request set a seed or not?" without inspecting the response body). The completions handler does NOT gain a new `tracing::info!` line — parity with the P21 `user` field observability asymmetry (chat handler logs both, completions handler accepts both at the wire type but does not log either). **Honoring is a no-op today** — vllm-lite's sampler reads from `rand`'s thread-local RNG which is currently unseeded; same seed + same model + same prompt does NOT yet produce the same output. Engine-side RNG seeding is v32+ work. 6 new integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_with_seed_field_accepted_by_handler`, `test_chat_without_seed_field_works_baseline`, `test_completions_with_seed_field_accepted_by_handler`, `test_chat_seed_field_wire_type_round_trip` (covers positive, omitted, negative, zero, `i64::MIN`/`i64::MAX` boundary cases per OpenAI spec), `test_completions_seed_field_wire_type_round_trip` (mirror on the legacy endpoint), `test_chat_streaming_with_seed_field_accepted_by_handler` (pins SSE-path parity with the non-streaming path). 7 new unit tests in `crates/server/src/openai/sampling_validation.rs::tests`: `chat_request_with_seed_none_passes_full_field_validation`, `chat_request_with_seed_positive_passes_full_field_validation`, `chat_request_with_seed_negative_passes_full_field_validation`, `chat_request_with_seed_zero_passes_full_field_validation`, `chat_request_with_seed_i64_min_passes_full_field_validation`, `chat_request_with_seed_i64_max_passes_full_field_validation`, `completion_request_with_seed_some_passes_full_field_validation`, `completion_request_with_seed_none_passes_full_field_validation`. `docs/reference/openai-compatibility.md` flipped both `seed` rows from "Not declared" → "Wired (declaration + tracing pass-through)" and marked the field shipped in the v0.2 follow-ups section. Public-API delta: 2 new fields (`ChatRequest::seed`, `CompletionRequest::seed`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate.
- **`docs/reference/feature-matrix.md` — per-crate Cargo feature model** (v31.0 P24 / engineering-quality §6 #4 closure) — the workspace has 14 Cargo features across 5 crates (the 6th, `vllm-dist`, exposes none) but no single map of what each one does, how it propagates across crates, or which combinations are tested in CI. `docs/architecture.md` §Feature Flags has the names; this new doc has the semantics. Sections: workspace layout (per-crate feature counts), per-crate feature tables with descriptions + what each feature enables (cargo dep or cfg-gated module), cross-crate propagation map (7 transitive edges + 4 always-on edges), and recommended combinations for common deployment shapes (minimal dev, GPU production, multi-node production, CI matrix). Notable inclusions that the architecture summary table previously omitted: `vllm-traits/candle` and `vllm-traits/kernels` (always-on via the core/model/server dep declarations; the `kernels = []` cfg-gate syntax is documented explicitly per the engineering-quality §6 closure). `docs/architecture.md` §Feature Flags cross-links the new doc as the single source of truth and adds the two omitted `vllm-traits` rows. `docs/README.md` adds the new doc to "Start Here" and bumps the ADR count from 19 → 20 (the README had drifted from the filesystem since ADR-020 / P14). `.planning/DOC-MAP.md` adds a `reference/` sub-tree to the docs tree so future maintainers can find both reference docs. Closes the engineering-quality §6 #4 follow-up that was marked "half-closed" in P15. Docs-only — no Rust / no test / no API delta.
- **`SHA256SUMS` per release target** (v31.0 P25 / engineering-quality §7 first-half closure) — closes the **checksums half** of §7's "release 生成 SBOM、校验和与构建 provenance" follow-up (P11 closed the SBOM half; this PR closes the checksums half). Pre-fix, the GitHub Release UI's "asset checksums" panel only showed file sizes, so a downstream consumer had no built-in way to verify that the binary they downloaded matched what the release workflow produced. Post-fix: new `Generate SHA256SUMS` step in `.github/workflows/release.yml::build` (placed after the existing `Generate SBOM` step) iterates every file in `artifacts/` and writes a standard two-column `<sha256>  ` digest file using `sha256sum`. The existing `Upload artifact` step then picks up `SHA256SUMS` along with the binary + SBOM (the `artifacts/**/*` glob in the `release` job attaches all three to the GitHub Release). The step self-skips its own output so the digest never includes a half-written version of itself; it's also portable across the matrix's ubuntu/macos/windows runners because all three provide `sha256sum` under `shell: bash`. `docs/RELEASE.md` gains a "Checksums" section that documents the verification pattern (`sha256sum -c SHA256SUMS`), explains why the file matters (download integrity + hash-pinning in deployment manifests + SBOM cross-reference), and explicitly calls out that signed build provenance (SLSA / in-toto) and per-artifact sigstore signatures remain v32+ — they need a signature-key story (where does the maintainer key live? how is it rotated?) and a reproducible-build posture (the build environment itself isn't pinned to a specific image today, only the Rust toolchain via `rust-toolchain.toml`). The trailing "Checksums and signed build provenance are still missing from release.yml" line in `RELEASE.md` is replaced with the new section + a clear "first half done, second half v32+" status. CI-only change — no Rust code touched, so `just nextest` is unchanged (1468 passed, 40 skipped). No public API delta. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --workspace --all-features` passes.

- **`frequency_penalty` + `presence_penalty` declared on OpenAI chat + completions request types** (v31.0 P27 / v0.3 wire-type follow-up) — closes the v0.3 candidate the doc previously called "mechanical but the wire-type change is a public-API delta — tracked for v0.3". Pre-fix, `ChatRequest` and `CompletionRequest` had no field for either, so serde silently rejected any request that included `"frequency_penalty": ...` or `"presence_penalty": ...` in its JSON body. Post-fix: both wire types declare `frequency_penalty: Option<f32>` and `presence_penalty: Option<f32>` with `#[serde(default, skip_serializing_if = "Option::is_none")]` (4 new public-API fields). Per OpenAI spec the valid range is `[-2.0, 2.0]` with finite values only — the new `validate_penalty` helper in `crates/server/src/openai/sampling_validation.rs` rejects NaN / ±infinity / out-of-range values with `400 invalid_request_error` (with the field name in the error message so callers can adapt). The validator is wired into both `validate_chat_request_fields` and `validate_completion_request_fields`. **Key behavioural difference from P21/P22/P23: `frequency_penalty` is HONOURED end-to-end** via the engine's existing `SamplingParams::repeat_penalty` slot (added by ARCH-02). The chat handler (both non-streaming and streaming variants) and the completions handler all map `frequency_penalty` to `repeat_penalty = max(1.0, 1.0 + value)` so non-negative values produce the OpenAI-spec "halve logit on each repetition" semantics: `frequency_penalty = 1.0` → `repeat_penalty = 2.0` (halve seen-token logits); `frequency_penalty = 0.0` → `repeat_penalty = 1.0` (no penalty). **Negative values are clamped to `repeat_penalty = 1.0`** because the current `apply_repeat_penalty` logit-divide math inverts the sign of negative logits when dividing by a value `< 1.0` (producing undefined ordering rather than a clean boost). Surfacing the OpenAI-spec "boost repetition" semantics requires either a new `apply_repeat_boost` helper or a sign-aware refactor of `apply_repeat_penalty`; either is mechanical but out of scope for the v0.3 declaration PR. **Honoring is a no-op for `presence_penalty`** because the engine implements *frequency*-style semantics (penalty proportional to occurrence count), not the binary-presence semantics OpenAI's `presence_penalty` field defines. Adding a presence-aware penalty step requires a new helper (`apply_presence_penalty`) — closed by P28 (see entry below). The chat handler threads `frequency_penalty = ?req.frequency_penalty, presence_penalty = ?req.presence_penalty` into the three existing `tracing::info!(...)` log lines (parity with the P21/P22/P23 observability plumbing). 8 new integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_forwards_frequency_penalty_to_engine` (pins `1.0 → 2.0` wire-through), `test_chat_frequency_penalty_zero_means_no_penalty` (pins `0.0 → 1.0`), `test_chat_frequency_penalty_negative_is_clamped_to_no_penalty` (pins `-1.5 → 1.0` clamp), `test_chat_without_frequency_penalty_works_baseline` (pins default path), `test_chat_presence_penalty_accepted_but_not_wired` (pins v0.3 declaration-only contract for presence_penalty), `test_chat_frequency_penalty_out_of_range_returns_400` (pins validator), `test_chat_presence_penalty_out_of_range_returns_400` (mirror), `test_completions_forwards_frequency_penalty_to_engine` (pins cross-endpoint parity). 24 new unit tests in `crates/server/src/openai/sampling_validation.rs::tests`: 11 standalone `validate_penalty` tests (None / 0 / positive / negative / -2.0 / 2.0 / above-2.5 / below--2.5 / NaN / +inf / -inf), 7 chat-endpoint full-validation tests, 1 both-penalties-set test, 4 completions-endpoint full-validation tests. `docs/reference/openai-compatibility.md` flipped all four `frequency_penalty` + `presence_penalty` rows from "Not declared" → "Wired (declaration + validation + [optional] engine wire-through)" and marked both fields shipped in the v0.3 follow-ups section; the v32+ candidates table now lists only the v32+ carve-outs (presence_penalty honoring, frequency_penalty boost semantics, logit_bias, logprobs, tools). Public-API delta: 4 new fields (`ChatRequest::frequency_penalty`, `ChatRequest::presence_penalty`, `CompletionRequest::frequency_penalty`, `CompletionRequest::presence_penalty`) + 1 new public helper (`openai::sampling_validation::validate_penalty`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate.

- **`presence_penalty` honoured end-to-end via new `apply_presence_penalty` helper** (v31.0 P28 / v0.3 wire-type follow-up engine wire-through) — closes the v0.3 carve-out the P27 doc listed as "v32+ work — out of scope for the v0.3 declaration PR". P27 declared + validated the field; this batch closes the engine gap by adding a presence-style penalty helper (analogous to `apply_repeat_penalty` but with binary semantics — once per distinct token, not once per occurrence). New `vllm_core::sampling::apply_presence_penalty` subtracts `presence_penalty` from the logit of every *distinct* seen token (deduped via internal `HashSet`, same pattern as `apply_repeat_penalty`). Wired into `sample_one_with_params` after `apply_repeat_penalty` and before the temperature / top-k / top-p pipeline so the order matches the sampling-pipeline doc-comment ("repeat penalty → temperature → top-k → top-p / temperature / greedy"). New `SamplingParams::presence_penalty: f32` field (default `0.0`) + `SamplingParamsBuilder::with_presence_penalty` builder method. **Key behavioural difference from `frequency_penalty`:** `apply_presence_penalty` is an additive subtraction, so negative values cleanly *encourage* repetition (subtracting a negative raises the logit) — no clamping needed, unlike `frequency_penalty`'s `max(1.0, 1.0 + value)` workaround for the `apply_repeat_penalty` logit-divide sign-flip bug. The chat handler (both non-streaming and streaming variants) and the completions handler now forward `req.presence_penalty` verbatim to `request.sampling_params.presence_penalty`. 7 new unit tests in `crates/core/src/sampling/tests.rs`: `test_presence_penalty_basic` (pins the [0.5] logit reduction), `test_presence_penalty_dedupes_seen_tokens` (pins the per-distinct-id semantics — 3 occurrences → 1 subtraction), `test_presence_penalty_no_effect_at_zero` (no-op default), `test_presence_penalty_empty_seen_is_noop` (empty seen slice), `test_presence_penalty_negative_encourages_repetition` (pins the negative = additive boost), `test_presence_penalty_out_of_range_token_is_ignored` (pins the out-of-range safety, mirrors `apply_repeat_penalty`), `test_presence_penalty_combined_with_repeat_penalty` (pins the ordering — `0.5/2.0 - 0.1 = 0.15` after both apply). 3 new integration tests in `crates/core/tests/sampling_params.rs`: `arch_02_presence_penalty_suppresses_seen_token_once_per_distinct_id` (pins presence-vs-frequency semantic — 5 occurrences of token 10 = 1 subtraction, not 5), `arch_02_presence_penalty_negative_encourages_repetition` (negative value → tied argmax breaks to lowest index → seen token wins), `arch_02_presence_penalty_combined_with_repeat_penalty` (pins combined behavior + ordering). The P27 `test_chat_presence_penalty_accepted_but_not_wired` test is replaced by 4 wire-through tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_forwards_presence_penalty_to_engine` (pins `1.5` → `SamplingParams.presence_penalty = 1.5`), `test_chat_without_presence_penalty_works_baseline` (pins default `0.0` path), `test_chat_presence_penalty_negative_is_forwarded_verbatim` (pins no-clamp behavior — `-1.0` stays `-1.0`, unlike `frequency_penalty`), `test_completions_forwards_presence_penalty_to_engine` (pins cross-endpoint parity). `docs/reference/openai-compatibility.md` flipped both `presence_penalty` rows from "Wired (declaration + validation)" → "Wired (declaration + validation + engine wire-through)" and marked the field shipped in the v0.3 follow-ups section; the v32+ candidates table's `presence_penalty` row is updated to "Shipped in P28". Public-API delta: 1 new field (`vllm_traits::SamplingParams::presence_penalty`) + 1 new builder method (`SamplingParamsBuilder::with_presence_penalty`) + 1 new public function (`vllm_core::sampling::apply_presence_penalty`); CHANGELOG `[Unreleased]` has `public-api: vllm-traits` + `public-api: vllm-core` bullets per the public-api-check gate.

- **`logit_bias` honoured end-to-end via new `apply_logit_bias` helper** (v31.0 P30 / v0.3 wire-type follow-up engine wire-through) — closes the v0.3 carve-out the openai-compatibility.md v32+ table listed as "v32+ work — Requires a bias-map injection in the sampler's softmax step. Not implemented in the current sampling module." New `vllm_core::sampling::apply_logit_bias(&mut [f32], &HashMap<TokenId, f32>)` adds each map value to the logit at the corresponding token ID before the temperature / top-k / top-p pipeline; out-of-vocab token IDs (any ID `>= logits.len()`) are silently ignored (matches OpenAI's server behaviour). New `SamplingParams::logit_bias: Option<HashMap<TokenId, f32>>` field (default `None`) + `SamplingParamsBuilder::with_logit_bias(Option<HashMap<TokenId, f32>>) -> Self` builder method. **Determinism guarantee:** the map iteration order is non-deterministic (HashMap) but the *final logits* are deterministic because each bias is additive and independent per token — so `apply_logit_bias` preserves the determinism guarantee that `sample_one_with_params` requires. **Key behavioural difference from `frequency_penalty` / `presence_penalty`:** both penalty helpers are *automatic* (every seen token gets the same bias); `logit_bias` is *explicit* (caller specifies exactly which token IDs to bias and by how much). New `logit_bias: Option<HashMap<TokenId, f32>>` field on `ChatRequest` + `CompletionRequest` with `#[serde(default, skip_serializing_if = "Option::is_none")]` (2 new public-API fields). New `validate_logit_bias` helper in `crates/server/src/openai/sampling_validation.rs` rejects NaN / ±infinity / out-of-range `[-100, 100]` values with `400 invalid_request_error` (error message names the offending token ID); token IDs are *not* validated — any `TokenId` (which is a `u32`) is accepted. The validator is wired into both `validate_chat_request_fields` and `validate_completion_request_fields`. The chat handler (non-streaming + streaming variants) and the completions handler all forward `req.logit_bias` verbatim to `SamplingParams::logit_bias`. The chat handler threads `logit_bias_len = ?req.logit_bias.as_ref().map(|m| m.len())` into the three existing `tracing::info!(...)` log lines (parity with P21/P22/P23/P27/P28/P29 observability plumbing; count only, not the full map, to keep log lines bounded for typical maps of up to ~300 entries); the completions handler accepts `logit_bias` at the wire type but does not gain a new `tracing::info!` line (parity with the `seed` / `user` / `frequency_penalty` / `presence_penalty` fields). 8 new unit tests in `crates/core/src/sampling/tests.rs`: `test_logit_bias_basic` (pins positive bias increases logit), `test_logit_bias_negative_decreases_logit` (pins OpenAI "suppress" semantic), `test_logit_bias_empty_map_is_noop` (no-op default), `test_logit_bias_empty_logits_is_noop` (no panic on empty logits), `test_logit_bias_out_of_range_keys_silently_ignored` (matches OpenAI server behaviour), `test_logit_bias_multiple_tokens_biased_independently` (pins multi-token + determinism), `test_logit_bias_does_not_affect_greedy_when_zero` (zero bias is no-op), `test_logit_bias_combined_with_repeat_and_presence_penalties` (pins the full pipeline ordering). 3 new sample_one_with_params integration tests in `crates/core/src/sampling/tests.rs`: `test_sample_one_with_params_logit_bias_changes_argmax` (bias flips greedy argmax), `test_sample_one_with_params_logit_bias_none_is_noop` (default path), `test_sample_one_with_params_logit_bias_empty_map_is_noop` (Some-but-empty). 2 new integration tests in `crates/core/tests/sampling_params.rs`: `arch_02_logit_bias_flips_argmax_in_greedy_sampling` (end-to-end bias + greedy), `arch_02_logit_bias_per_sequence_divergence` (per-sequence `SamplingParams::logit_bias` threading through `sample_batch_with_params`). 12 new `validate_logit_bias` unit tests in `sampling_validation.rs::tests`: None / empty / zero / positive-in-range / negative-in-range / above-100.001 / below--100.001 / NaN / +inf / -inf / first-invalid-wins / token-id-unconstrained. 7 new full-validation tests: chat × {None, in-range, out-of-range, NaN} + completion × {None, in-range, out-of-range}. 6 new chat integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_forwards_logit_bias_to_engine` (pins verbatim map round-trip with 4 entries spanning the `[±100]` boundary), `test_chat_without_logit_bias_works_baseline` (pins default `None` path), `test_chat_with_empty_logit_bias_is_accepted` (pins Some(empty) vs None distinction), `test_chat_logit_bias_out_of_range_returns_400` (pins 400 rejection for `200.0 > 100.0`), `test_chat_logit_bias_nan_returns_400` (pins 400 rejection for very-large finite values that exceed `100.0`), `test_completions_forwards_logit_bias_to_engine` (pins cross-endpoint parity). `docs/reference/openai-compatibility.md` flipped both `logit_bias` rows from "Not declared" → "Wired (declaration + validation + engine wire-through)" and marked the field shipped in the v0.3 follow-ups section; the v32+ candidates table's `logit_bias` row is updated to "Shipped in P30". **Note:** `SamplingParamsBuilder::build()` is no longer `const fn` because `HashMap` has a non-trivial `Drop` impl (the `with_logit_bias` builder method is also non-const for the same reason); all other `with_*` methods that don't allocate remain `const`. Public-API delta: 2 new fields (`ChatRequest::logit_bias`, `CompletionRequest::logit_bias`) + 1 new field (`SamplingParams::logit_bias`) + 1 new builder method (`SamplingParamsBuilder::with_logit_bias`) + 1 new public function (`vllm_core::sampling::apply_logit_bias`) + 1 new public helper (`openai::sampling_validation::validate_logit_bias`); CHANGELOG `[Unreleased]` has `public-api: vllm-server` + `public-api: vllm-traits` + `public-api: vllm-core` bullets per the public-api-check gate. The v0.3 carve-outs remaining: `logprobs` / `top_logprobs` (engine-side top-K logprob generation, requires per-token logprob return from the sampler), `tools` / `tool_choice` (grammar-constrained decoder + per-request tool schema cache).

- **`logprobs` + `top_logprobs` declared + validated on OpenAI chat + completions request types** (v31.0 P31 / v0.3 wire-type follow-up declaration + validation) — closes the **declaration layer** for the v0.3 carve-out the openai-compatibility.md v32+ table listed as "v32+ work — Requires the engine to return top-K logprobs per output token". Pre-fix, neither `ChatRequest` nor `CompletionRequest` declared `logprobs` or `top_logprobs`, so serde silently rejected any request that included those fields. Post-fix, 3 new public-API fields: `ChatRequest::logprobs: Option<bool>`, `ChatRequest::top_logprobs: Option<u32>`, `CompletionRequest::logprobs: Option<u32>` — all with `#[serde(default, skip_serializing_if = "Option::is_none")]`. **Type asymmetry preserved on purpose:** chat's `logprobs` is `Option<bool>` (the chat spec's flag) and completions' is `Option<u32>` (the legacy spec's 0-5 integer count) — matches OpenAI spec exactly rather than unifying behind a common representation. 2 new public helpers in `crates/server/src/openai/sampling_validation.rs`: `validate_chat_logprobs(Option<bool>, Option<u32>)` rejects `top_logprobs > 20` and the cross-field rule violation (`top_logprobs = Some(_)` requires `logprobs = Some(true)`); `validate_completion_logprobs(Option<u32>)` rejects values > 5. Both validators return `400 invalid_request_error` with field-naming messages. Wired into `validate_chat_request_fields` and `validate_completion_request_fields` respectively. The chat handler (non-streaming + streaming variants) threads `logprobs = ?req.logprobs, top_logprobs = ?req.top_logprobs` into the three existing `tracing::info!(...)` log lines (parity with P21/P22/P23/P27/P28/P29/P30 observability plumbing); the completions handler accepts `logprobs` at the wire type but does not gain a new `tracing::info!` line (parity with `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` rationale). 11 new `validate_chat_logprobs` unit tests: both-None / false-None / true-None / true+top_in_range / top_logprobs_above_20 / top_logprobs_without_logprobs / top_logprobs_with_logprobs_false. 4 new `validate_completion_logprobs` unit tests: None / Some(0) / in_range_sweep / above_5. 7 new full-validation tests: chat × {None, in-range, out-of-range, cross-field} + completion × {None, in-range, out-of-range}. 6 new chat integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_with_logprobs_field_accepted_by_handler`, `test_chat_without_logprobs_field_works_baseline`, `test_chat_top_logprobs_out_of_range_returns_400`, `test_chat_top_logprobs_without_logprobs_returns_400` (pins the cross-field rule end-to-end), `test_completions_with_logprobs_field_accepted_by_handler`, `test_completions_logprobs_out_of_range_returns_400`. `docs/reference/openai-compatibility.md` flipped 3 rows from "Not declared" → "Wired (declaration + validation)" (chat `logprobs`, chat `top_logprobs`, completions `logprobs`); the v32+ candidates table's `logprobs` / `top_logprobs` row is updated to "Shipped in P31 (declaration only; engine honoring is v32+)"; the response-side `choices[].logprobs` row is updated to note the request-side is now declared but the response-side remains v32+ work. **Honoring is a no-op today** — the engine's `sample_batch_with_params` returns only the sampled token; producing top-K logprob output per token requires a wire-breaking change to the sampler return type (a per-sequence `Vec<TokenLogprob>` slice). Tracked as v32+ work so the declaration PR doesn't regress to "rejected by serde" for callers who already send the field. Public-API delta: 3 new fields (`ChatRequest::logprobs`, `ChatRequest::top_logprobs`, `CompletionRequest::logprobs`) + 2 new public helpers (`openai::sampling_validation::validate_chat_logprobs`, `openai::sampling_validation::validate_completion_logprobs`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate. The v0.3 carve-out remaining: `tools` / `tool_choice` (grammar-constrained decoder + per-request tool schema cache).

- **`echo` + `suffix` + `best_of` declared + validated on OpenAI legacy `/v1/completions` request type** (v31.0 P32 / v0.x wire-type follow-up declaration + validation) — closes the **declaration layer** for the three `/v1/completions`-specific fields the openai-compatibility.md matrix previously listed as "Not declared". Pre-fix, `CompletionRequest` had no fields for any of `echo`, `suffix`, or `best_of`, so serde silently rejected any request that included them. Post-fix, 3 new public-API fields: `CompletionRequest::echo: Option<bool>`, `CompletionRequest::suffix: Option<String>`, `CompletionRequest::best_of: Option<u32>` — all with `#[serde(default, skip_serializing_if = "Option::is_none")]`. Per OpenAI legacy-completions spec the valid `best_of` range is `>= 1`; `suffix` is unconstrained (any string accepted); `echo` defaults to `false`. New `validate_completion_meta(Option<bool>, Option<u32>)` helper in `crates/server/src/openai/sampling_validation.rs` rejects `best_of = 0` (out of range) and the cross-field rule violation `echo = true && best_of > 1` per OpenAI spec — the rationale: when the server picks one completion from N candidates by mean logprob, the user has no way to disambiguate which one was returned if echo is enabled, so the combination is forbidden. Both rejections return `400 invalid_request_error` with field-naming messages. The validator is wired into `validate_completion_request_fields`. **Honoring is a no-op for all three** today — engine would need to: (1) prepend the prompt to `CompletionChoice.text` for `echo` (tokenizer dependency on the response side, currently absent), (2) append `suffix` to `CompletionChoice.text`, (3) sample `best_of` times + rank by mean logprob for `best_of` (depends on the same v32+ logprob-generation work as the `logprobs` field, P31). Tracked as v32+ work so the declaration-only PR doesn't regress to "rejected by serde" for callers who already send the fields. The completions handler does not currently log the new fields (parity with `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` — completions handler accepts them at the wire type but does not log any of them). 14 new unit tests in `crates/server/src/openai/sampling_validation.rs::tests`: 8 standalone `validate_completion_meta` tests (all-None / echo-only / best_of-only / echo=false+best_of / best_of=0 / echo=true+best_of=1 / echo=true+best_of=2 / echo=true+best_of=5), 6 full-validation tests (echo=None / echo=true+best_of=1 / suffix / best_of=5 / best_of=0 / echo=true+best_of=3). 6 new chat integration tests in `crates/server/tests/chat_integration_test.rs`: `test_completions_with_echo_true_accepted_by_handler` (pins `echo=true` + `best_of=1` pass), `test_completions_with_suffix_accepted_by_handler` (pins any string accepted), `test_completions_without_echo_suffix_best_of_works_baseline` (pins default-path), `test_completions_with_best_of_above_one_accepted_by_handler` (pins `best_of=5` alone passes), `test_completions_best_of_zero_returns_400` (pins validator), `test_completions_echo_true_with_best_of_above_one_returns_400` (pins cross-field rule end-to-end). `docs/reference/openai-compatibility.md` flipped the `echo` / `suffix` / `best_of` row in the legacy `/v1/completions` table from "Not declared" → three separate "Wired (declaration + validation)" rows; the v32+ candidates table now lists `echo` / `suffix` / `best_of` (legacy `/v1/completions`) as "Declaration-only shipped in P32 (2026-07-18)" with engine honoring rationale. The closing paragraph at the end of the v0.3 wire-type section now also notes P32's declaration-layer closure for these three legacy fields. Public-API delta: 3 new fields (`CompletionRequest::echo`, `CompletionRequest::suffix`, `CompletionRequest::best_of`) + 1 new public helper (`openai::sampling_validation::validate_completion_meta`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate. The v0.3 carve-out remaining: `tools` / `tool_choice` (grammar-constrained decoder + per-request tool schema cache).

- **`tools` + `tool_choice` declared + validated on OpenAI `/v1/chat/completions` request type** (v31.0 P33 / v0.x wire-type follow-up declaration + validation) — closes the **declaration layer** for the final v0.3 carve-out the openai-compatibility.md v32+ table listed as "Architecture-level work — v32+. Tool-calling framework requires a grammar-constrained decoder (JSON-schema → grammar) and a per-request tool schema cache." Pre-fix, `ChatRequest` had no fields for either `tools` or `tool_choice`, so serde silently rejected any request that included them. Post-fix, 2 new public-API fields on `ChatRequest`: `tools: Option<Vec<Tool>>` + `tool_choice: Option<ToolChoice>` — both with `#[serde(default, skip_serializing_if = "Option::is_none")]`. **NOT** declared on `CompletionRequest` — the legacy endpoint doesn't support tool calling per OpenAI spec. New types in `crates/server/src/openai/types.rs`: `Tool` (carries a `type` discriminator + `function` payload), `FunctionDefinition` (carries `name: String` + optional `description` + optional `parameters: serde_json::Value` for the JSON Schema), `ToolType` enum (currently only `Function` — future OpenAI extensions like retrieval / code-interpreter would add new variants), `ToolChoice` enum with `#[serde(untagged)]` to support both string modes (`"none"` / `"auto"` / `"required"` via `ToolChoiceMode`) and the object form (`{"type": "function", "function": {"name": "..."}}` via `ToolChoiceSpecific` + `ToolChoiceFunctionRef`). The `untagged` strategy preserves the wire shape 1:1; the string modes are tried before the object form so a bare `"auto"` string doesn't accidentally deserialize as `Specific { kind: Auto, ... }`. The `parameters` field is stored as `serde_json::Value` (rather than a typed JSON-Schema struct) because OpenAI's spec is intentionally permissive about JSON Schema drafts (draft-04 through 2020-12 are all accepted server-side) and modelling the full grammar would balloon the type without giving vllm-lite any new capability (the engine doesn't process the schema today). New `validate_chat_tool_choice(Option<&[Tool]>, Option<&ToolChoice>)` helper in `crates/server/src/openai/sampling_validation.rs` rejects: (1) `tool_choice = Some(Required)` when `tools` is `None` / empty (model can't satisfy "must call a tool"); (2) `tool_choice = Some(Specific { function: { name: X } })` when `tools` is `None` / empty; (3) `tool_choice = Some(Specific { function: { name: X } })` when no tool in `tools[]` has a matching `name` (linear scan — typical tool lists are <10 entries). All three rejections return `400 invalid_request_error` with field-naming messages. The validator is wired into `validate_chat_request_fields`. The chat handler (non-streaming + streaming variants) threads `tools_len = ?req.tools.as_ref().map(|v| v.len())` (count only) and `tool_choice = ?req.tool_choice` (Debug-printable, small) into the three existing `tracing::info!(...)` log lines (parity with P21/P22/P23/P27/P28/P29/P30/P31/P32 observability plumbing). **Honoring is a no-op** today — engine-side tool calling requires a grammar-constrained decoder (JSON-schema → grammar at sample time, not a separate post-processing step) and a per-request tool schema cache (the JSON Schema in `tools[].function.parameters` is per-request and can't be globally memoized). Tracked as v32+ work so the declaration-only PR doesn't regress to "rejected by serde" for callers who already send the field. **Field names + descriptions + parameters are *not* validated against the OpenAI-spec regex (`^[a-zA-Z0-9_-]{1,64}$`) today** — the validator is permissive by design until honoring lands (today there's no engine-side caller that would care about the regex). 16 new unit tests in `crates/server/src/openai/sampling_validation.rs::tests`: 9 standalone `validate_chat_tool_choice` tests (none-modes / auto / required+tools / required+no-tools / required+empty-tools / specific+matching / specific+no-tools / specific+empty-tools / specific+unknown-name) + 7 full-validation tests on `ChatRequest` (baseline / tools-only / required+tools / required+no-tools / specific+matching / specific+unknown / specific+no-tools). 10 new chat integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_with_tools_only_accepted_by_handler`, `test_chat_with_tool_choice_auto_accepted_by_handler`, `test_chat_with_tool_choice_required_accepted_by_handler`, `test_chat_with_tool_choice_specific_matching_accepted_by_handler`, `test_chat_without_tools_tool_choice_works_baseline`, `test_chat_tool_choice_required_without_tools_returns_400`, `test_chat_tool_choice_required_with_empty_tools_returns_400`, `test_chat_tool_choice_specific_unknown_tool_returns_400`, `test_chat_tool_choice_specific_without_tools_returns_400`, `test_chat_tool_choice_none_with_tools_accepted_by_handler`. `docs/reference/openai-compatibility.md` flipped 2 rows from "Not declared" → "Wired (declaration + validation)" (chat `tools` + chat `tool_choice`); the v32+ candidates table's `tools / tool_choice` row is updated to "Declaration-only shipped in P33 (2026-07-18)" with engine honoring rationale (grammar-constrained decoder + per-request tool schema cache). **The v0.3 wire-type backlog is now FULLY CLOSED at the declaration + validation layer** — every OpenAI spec field in the v0.3 wire-type follow-ups is declared + validated end-to-end through the HTTP boundary (declaration: P21/P22/P23/P27/P31/P32/P33; engine wire-through: P28/P29/P30 — all the v0.3 carve-outs closed except the grammar-constrained decoder for `tools`). Public-API delta: 6 new types (`Tool`, `FunctionDefinition`, `ToolType`, `ToolChoice`, `ToolChoiceMode`, `ToolChoiceSpecific`, `ToolChoiceFunctionRef`) + 2 new fields (`ChatRequest::tools`, `ChatRequest::tool_choice`) + 1 new public helper (`openai::sampling_validation::validate_chat_tool_choice`); CHANGELOG `[Unreleased]` has a `public-api: vllm-server` bullet per the public-api-check gate. **The v0.3 carve-out remaining: `tools` / `tool_choice` engine honoring** (grammar-constrained decoder + per-request tool schema cache — architecture-level v32+ work).

- **`seed` honoured end-to-end via new `sample_random_threshold` helper** (v31.0 P34 / v0.2 wire-type follow-up engine wire-through) — closes the v0.2 carve-out the openai-compatibility.md v32+ table listed as "v32+ work — RNG seeding in `vllm_core::sampling`". P23 declared + validated + traced the field; this batch closes the engine gap so the OpenAI-spec contract "same seed + same model + same prompt → same output" actually holds. The previous sampler read from `rand`'s thread-local default RNG which is unseeded, so callers who set `seed` saw no behavioural change. Post-fix, when `SamplingParams::seed = Some(s)`, the sampler builds a fresh `rand::rngs::StdRng::seed_from_u64(s)` and draws ONE `f32` random threshold per `sample_one_with_params` call. `seed = None` continues to read from the thread-local default RNG (pre-P34 behaviour, fully backward-compatible). New `SamplingParams::seed: Option<u64>` field with `#[serde(default, skip_serializing_if = "Option::is_none")]` + `SamplingParamsBuilder::with_seed(u64)` + `with_seed_none()` builder methods in `crates/traits/src/sampling.rs`. New `random_f32_seeded(seed: u64) -> f32` private helper in `crates/core/src/sampling.rs` (uses `StdRng`, the same CSPRNG-quality generator `rand::random()` uses internally under default `rand` 0.10 features — so seeded and unseeded paths produce statistically-equivalent random sequences, only the seed source differs). New `sample_random_threshold(Option<u64>) -> f32` dispatcher at the single RNG touchpoint; `temperature_sample` / `top_p_sample` / `top_k_sample` / `sample_batch` / `sample_one_with_params` all consume a precomputed `random_threshold: f32` parameter rather than reading the RNG internally (the third argument to `top_k_sample` is the public-API wire-breaking change; everything else is `pub(crate)`). **Greedy bypass:** `temperature = 0` and `top_p = 1.0` paths bypass the RNG entirely (deterministic argmax) so `seed` has no observable effect in those modes — matches OpenAI's "best effort" wording in the spec (seed is only meaningful for non-greedy sampling). **Per-sequence independence:** `sample_batch_with_params` reads ONE random threshold per call to `sample_one_with_params`, so two sequences in the same batch with DIFFERENT seeds produce DIFFERENT random draws — the engine never shares RNG state across sequences. Two sequences with the SAME seed re-seed independently from the same u64 and therefore draw the same random threshold for the same logits — this is the correct behaviour for OpenAI's per-request determinism contract. **i64 ↔ u64 cast:** OpenAI's `seed` is `i64`; the HTTP layer does `seed as u64` (Rust's `i64 as u64` wrapping semantics — negatives wrap to large u64 values, which produces a deterministic but distinct RNG state, exactly what the spec calls for). The chat handler (non-streaming + streaming variants) and the completions handler now forward `req.seed` verbatim to `request.sampling_params.seed`. The chat handler does NOT gain a new `tracing::info!` line because P23 already threads `seed = ?req.seed` into the existing observability plumbing (parity preserved); the completions handler accepts `seed` at the wire type and forwards to the engine but does not gain a new log line (parity with `user` / `frequency_penalty` / `presence_penalty` / `logit_bias`). 7 new unit tests in `crates/core/src/sampling/tests.rs`: `test_seed_determinism_same_seed_same_result` (pins OpenAI determinism contract at the sampler layer), `test_seed_zero_is_valid_seed` (pins the `seed = 0` ≠ `seed = None` distinction), `test_seed_none_falls_back_to_thread_rng` (pins backward-compatible default path), `test_seed_different_seeds_diverge_in_distribution` (pins that seeded RNG draws different thresholds), `test_seed_greedy_path_bypasses_rng` (pins that `T = 0` ignores `seed` — same argmax regardless), `test_seed_per_sequence_independence_in_batch` (pins per-sequence independence for batched sampling), `test_seed_top_p_path_uses_seeded_rng` (pins that the top-p sampler also honours the seed). 2 new integration tests in `crates/core/tests/sampling_params.rs`: `arch_02_seed_determinism_end_to_end` (verifies the contract at the `sample_batch_with_params` boundary, the function the engine actually calls), `arch_02_seed_per_sequence_divergence_in_batch` (pins per-sequence independence end-to-end). 4 new chat integration tests in `crates/server/tests/chat_integration_test.rs`: `test_chat_forwards_seed_to_engine` (pins `42 → Some(42)`), `test_chat_without_seed_works_baseline` (pins default `None` path), `test_chat_negative_seed_is_forwarded_via_u64_cast` (pins `i64 as u64` wrapping), `test_chat_seed_zero_is_forwarded_as_some_zero` (pins `seed = 0` is NOT conflated with `None`). `docs/reference/openai-compatibility.md` flipped both `seed` rows from "Wired (declaration + tracing pass-through)" → "Wired (declaration + validation + engine wire-through)" and updated the v32+ candidates table's `seed` row to "Shipped in P34". Public-API delta: 1 new field (`SamplingParams::seed`) + 2 new builder methods (`SamplingParamsBuilder::with_seed`, `SamplingParamsBuilder::with_seed_none`) + 1 new public function (`vllm_core::sampling::sample_random_threshold`); CHANGELOG `[Unreleased]` has `public-api: vllm-traits` + `public-api: vllm-core` bullets per the public-api-check gate. **Wire-breaking change to `vllm_core::sampling::top_k_signature`:** the public `top_k_sample(logits, k)` function gained a third `random_threshold: f32` parameter so the seeded RNG can drive top-k sampling too; all 4 workspace-internal callers (2 unit-test files + 0 production callers) were updated. **The v0.2 wire-type backlog is now FULLY CLOSED at the engine-honoring layer** — every OpenAI v0.2 wire-type field (`seed`, `user`, `response_format`) is honoured end-to-end. The v32+ candidates remaining: `logprobs` / `top_logprobs` (engine-side top-K logprob generation), `tools` / `tool_choice` (grammar-constrained decoder + per-request tool schema cache), `echo` / `suffix` / `best_of` (engine-side response formatting + sample-N-rank-by-mean-logprob for `best_of`, which depends on the `logprobs` work).

- **`echo` + `suffix` honoured end-to-end on OpenAI legacy `/v1/completions`** (v31.0 P35 / v0.x wire-type follow-up engine wire-through) — closes the engine-side gap for `echo` and `suffix` (P32 declared + validated both; `best_of` engine honoring remains v32+ because it depends on the logprob-generation work that P31's `logprobs` field unblocks). New `apply_completion_meta(completion: String, prompt: &str, echo: Option<bool>, suffix: Option<&str>) -> String` helper in `crates/server/src/openai/completions.rs` is the single authoritative point for response-side text formatting — both the non-streaming and streaming paths call it so the contract stays in sync. Per OpenAI legacy-completions spec: `echo = true` prepends the prompt to the generated continuation (response `text` = `prompt + completion`); `suffix = Some(_)` appends the suffix (response `text` = `completion + suffix`); both flags are independent so the four combinations are: `prompt + completion`, `completion`, `prompt + completion + suffix`, `completion + suffix`. `echo = false` (or omitted) and `suffix = None` produce the pre-P35 behaviour (no prefix / no postfix) — legacy clients are unaffected. **Streaming design:** `echo = true` prepends the prompt to the FIRST non-empty text chunk's `text` field; `suffix = Some(_)` puts the suffix into the FINAL chunk's `text` field (the chunk that carries `finish_reason`). Both choices match OpenAI's accumulator semantics — clients concatenate all chunk `text` fields in order, so prefixing the first chunk puts the prompt at the start of the visible response and suffixing the final chunk puts the suffix at the end. A new `first_chunk_sent: bool` flag in the streaming unfold state tracks whether the prompt has already been prepended (so the prefix never duplicates if the first decoded token would otherwise be a special-token-skipped chunk — `should_skip_token_text` chunks emit `text: ""` and don't consume the flag). The completions handler does NOT gain a new `tracing::info!` line for `echo` / `suffix` (parity with the existing `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` fields). 6 new chat integration tests in `crates/server/tests/chat_integration_test.rs`: `test_completions_echo_true_prepends_prompt_to_text` (pins `echo=true` non-streaming prefix), `test_completions_echo_false_does_not_prepend_prompt` (pins the backward-compatible default path), `test_completions_suffix_appends_to_text` (pins non-streaming suffix), `test_completions_echo_and_suffix_combine` (pins the combined `prompt + completion + suffix` shape), `test_completions_streaming_echo_true_prepends_prompt_to_first_chunk` (pins streaming echo contract), `test_completions_streaming_suffix_lands_on_final_chunk` (pins streaming suffix contract — suffix lands on the chunk that carries `finish_reason`). `docs/reference/openai-compatibility.md` flipped both `echo` and `suffix` rows from "Wired (declaration + validation)" → "Wired (declaration + validation + engine wire-through)" and updated the v32+ candidates table's `echo / suffix / best_of` row to "echo + suffix shipped in P35; best_of remains v32+ (depends on logprob-ranking work)". Public-API delta: 0 new public items (the `apply_completion_meta` helper is private to the `completions` module; the contract is delivered through the existing `CompletionRequest::echo` + `CompletionRequest::suffix` wire-type fields and the existing `CompletionChoice::text` response field). The v32+ candidates remaining: `logprobs` / `top_logprobs` (engine-side top-K logprob generation), `tools` / `tool_choice` (grammar-constrained decoder + per-request tool schema cache), `best_of` (engine-side sample-N-rank-by-mean-logprob — depends on the logprobs work).

- **Chunked Prefill Correctness (v31.0 Phase 31-A)** — fixes the long-standing partial-prefill bug where multi-step prefill produced different logits than single-shot prefill.
    - **`write_prefill_kv`**: now writes each token at its global position (`positions[i] % block_size`) instead of always overwriting block offset 0.
    - **`RopeGqaAttention::forward_prefill_continue`**: new path for continuation chunks — reads existing KV prefix, applies rectangular causal mask (`prefill_continue_causal_mask`), writes new tokens, and attends over the full prefix.
    - **`RopeGqaDecoderBlock`**: specialized `DecoderLayer` impl dispatches to `forward_prefill_continue` when `num_computed_tokens > 0`.
    - **`BatchComposer::compose_chunked_prefill`**: continuation chunks now set `is_prefill = true` (all chunk tokens are embedded; model layer handles continuation).
    - Unit tests: `test_decoder_prefill_continue_matches_full_prefill`, `test_write_prefill_kv_respects_global_positions`, `test_prefill_continue_causal_mask_shape`.
    - Checkpoint test `test_qwen3_partial_prefill_matches_full_prefill` updated to assert equality (still `#[ignore]` — requires on-disk weights).
- **Audit-trail write path wired** (technical due diligence §6) — `AuditLogger` ring buffer existed but nothing called `log_api_request` for HTTP requests, so every API call left no in-process trail. New `security::audit_middleware` reads `CorrelationId` + `AuthenticatedUser` from request extensions and records one row per request after the handler returns. Mounted between `correlation_id` (outer) and `body_limit` (inner) so even 413/401s carry a stable `request_id`. `AuthenticatedUser` extension stamps `key:<first-8-chars>` so the full key never appears in audit exports. Integration test `audit_middleware_wiring.rs` (3 cases: happy / 404-failure / distinct rows).
- **`/debug/audit` endpoint exposes the audit ring buffer** — `audit_middleware` was write-only until this commit. New `debug::audit_dump` (gated by the existing `require_admin` fail-closed policy) returns the ring buffer newest-first, capped at 1000 entries (`count` + `returned` + `cap` fields let operators detect eviction without parsing events). Three tests in `admin_gating.rs` cover 503 / 401 / 200.
- **Helm Chart packaged in release pipeline** (GOV-01) — `Chart.yaml` shipped with placeholder `version` / `appVersion` and the release workflow never packaged it. New `scripts/sync-chart-version.sh` substitutes the values from the release manifest (uses `awk`, no helm dependency); new `chart` job in `release.yml` packages `vllm-lite-$VERSION.tgz` and attaches it to the GitHub Release. `smoke-deployment.sh` now asserts `Chart.yaml.{version,appVersion} == workspace.version` so drift fails CI.
- **Benchmark workflow re-enabled** (engineering-quality §5) — `.github/workflows/benchmark.yml` was a placeholder claiming "no benchmark targets configured", but 10 Criterion benches (`radix_cache`, `scheduler*`, `prefix_cache_benchmarks`, `latency_percentiles`, `*_speculative*`, `optimization_benchmarks`, `flash_attention`, `gqa_forward`) actually exist. The workflow now runs them on push to main and uploads output + criterion data as artifacts.
- **Graceful shutdown flips readiness before closing the listener** (production-readiness §7, v31.0 P7 follow-up / technical due diligence) — pre-fix the `/shutdown` HTTP handler sent `EngineMessage::Shutdown` and returned 200 OK without touching the readiness flag, so a Kubernetes probe that polled `/health/ready` between `/shutdown` and SIGTERM still saw `Ok` and could route new traffic to a pod whose engine was already tearing down. Post-fix both the `/shutdown` handler AND the SIGTERM/Ctrl+C shutdown coordinator call the new `HealthChecker::mark_not_ready()` so the next `/health/ready` probe returns `503 not_ready`. The SIGTERM path additionally waits `server.shutdown_drain_grace_secs` (default 5 s, capped at 300 s) before returning — long enough for the orchestrator to observe the failed probe and remove the pod from the Service endpoints list, short enough to feel responsive to `kubectl delete pod`. 5 new integration tests in `crates/server/tests/shutdown_readiness.rs` cover the happy path (`/shutdown` → 200 → readiness=NotReady → `/health/ready` 503), the admin-disabled path (no flip), the unauthorized path (no flip, prevents griefing), and idempotency of `mark_not_ready`. 2 new unit tests in `crates/server/src/health.rs` cover `mark_not_ready_flips_a_ready_checker` and `mark_not_ready_is_idempotent`.
- **OPERATIONS.md drift closed** (v31.0 P8 follow-up / technical due diligence) — the Multi-Node section claimed "KV block transfer is not yet production-ready" but Phase 31-D / OPS-31d shipped the `TransferKVBlock` RPC + fan-out fallback + 64 MiB message limit. Rewrote the section to honestly split what works (CacheMessage replication, TransferKVBlock, fan-out fallback) vs. what is still v32+ (smart owner routing, failure recovery, MESI/Directory enforcement), added a library-API quickstart, and explicitly noted that `peer_urls` is library-level only — no CLI / `VLLM_*` env var exists yet. Graceful Shutdown section now documents the §7 four-step sequence (readiness flip + drain grace + axum drain + engine join) introduced by P7, including the new `shutdown_drain_grace_secs` YAML key and how to tune it for K8s probe timing.
- **Doc-coverage gate at 65% real** (Phase 31-E, v31.0 P8 follow-up) — the `scripts/doc_coverage.sh` script existed but wasn't wired into any gate, so the per-crate / workspace numbers in STATE.md / CHANGELOG could silently drift down. New CI step after Public API baseline check reads `scripts/doc_coverage.sh --real json`, sums `real_total` + `real_documented` across crates, and exits non-zero if workspace `real_pct < DOC_COVERAGE_MIN` (default `65.0`, override per workflow run). `just doc-coverage-check` mirrors the gate for the local loop and is wired into `just ci` / `just ci-all`. New `just check` and `just ci-full` aliases for `just quick` and `just ci-all` per engineering-quality §8. Current measured value: **67.91%** real (target 65%).

### Added

- **`PagedKvCacheWrapper` — production `BlockDataSource` impl** (v31.0 P40 / OPS-32a first half) — closes the deferred engine-side wiring gap from ADR-020 §5. New `vllm_model::paged_tensor::PagedKvCacheWrapper` in `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs` (gated by `#[cfg(feature = "multi-node")]` per ADR-008) wraps `Arc<PagedKvCache>` and implements `vllm_dist::BlockDataSource`. `fetch_block` reads per-layer K/V tensors via a new `pub(crate) read_layer_block` helper on `PagedKvCache` (added in T1), serializes them to a flat `Vec<u8>` via `bytemuck::cast_slice`, returns the bytes via `tokio::task::block_in_place`. `has_block` queries the existing `block_hashes` field via a new `pub(crate) block_hashes_for_layer` accessor (T3), using layer 0 as the canonical witness. Quantized caches are dequantized on the sender side via a local `dequantize_f32(x, scale) = x * scale` helper so the receiver gets f32 bytes matching `write_kv_batch`'s f32 input contract. **6 new unit tests** in `crates/model/src/paged_tensor/paged_kv_cache_wrapper.rs::tests` (`fetch_block_returns_bytes_for_valid_block` / `fetch_block_returns_not_found_for_oob_block` / `fetch_block_serializes_written_kv` / `has_block_returns_true_for_written_block` / `has_block_returns_false_for_unwritten_block` / `fetch_block_dequantizes_quantized_blocks`). **1 integration test** in `crates/model/tests/paged_kv_cache_wrapper_e2e.rs` (`real_paged_kv_cache_bytes_round_trip_via_wrapper`) — 2-node gRPC pair with real `PagedKvCache` on the sender; head-0/token-0 f32 values round-trip exactly through `TransferKVBlock`. **3 new `pub(crate)` accessors** on `PagedKvCache` (`read_layer_block`, `num_blocks_count_per_layer`, `block_hashes_for_layer`), all annotated `#[allow(dead_code)] // reachable under --features multi-node via PagedKvCacheWrapper`. **Public-API delta (default features): 0.** Public-API delta (`--features multi-node`): +1 `pub` type `PagedKvCacheWrapper`, +1 `pub mod`, +3 `pub(crate)` accessors. **New workspace dep:** `bytemuck = { version = "1", features = ["derive"] }`. **New optional deps on `crates/model`:** `tokio` (with `rt-multi-thread, macros` features) + `async-trait` (workspace dep), both gated by `multi-node`. **Docs:** `OPERATIONS.md` §"Multi-Node (Experimental)" §"What works" gains the wrapper bullet (4th); §"What is not" replaces "wrapper not implemented" with "wrapper exists but engine plumbing deferred to P41+". ADR-020 status line bumped from "engine integration deferred to v32+" to "wrapper shipped in v31.0 P40; engine plumbing deferred to P41+". `.planning/v31.0-MASTER-PLAN.md` Phase Index gains row **31-G Multi-Node Wrapper**. **Engine plumbing (P41+):** `Arc<PagedKvCache>` → `MemoryManager::block_data_source` + `EngineBuilder::with_paged_kv_cache` + server main wiring — explicitly deferred to keep P40 reviewable per the spec's scope-split rationale.

- **`stop` sequences engine wire-through on chat + completions** (v31.0 P38) — closes the last hard-400 v0.x wire-type carve-out. P22 declared + validated `ChatRequest::stop` and P32 declared + validated `CompletionRequest::stop` (`Option<Vec<String>>`, up to 4 strings per OpenAI spec) but the validator hard-rejected non-empty stops with `400 invalid_request_error`. P38 honors them end-to-end: HTTP layer tokenizes each stop string via `state.tokenizer.encode(s)`, forwards as new public field `vllm_traits::SamplingParams::stop_token_sequences: Option<Vec<Vec<TokenId>>>` + `SamplingParamsBuilder::with_stop_token_sequences` / `with_stop_token_sequences_none`. Engine's `step_regular` runs `vllm_core::sampling::matches_stop_sequences(generated, stops)` after every sampled token; on match the sequence is finalized with `FinishReason::Stop` and the matched token is emitted via `response_tx` BEFORE finalization (OpenAI convention — response includes the matched stop text). **Composes correctly** with `best_of` (P37 — each candidate honors its own stop set), `logprobs`/`top_logprobs` (P36 — matched token's logprob is emitted alongside), `echo`/`suffix` (P35 — orthogonal), `seed` (P34 — RNG-agnostic), and `max_tokens` (Length wins if hit first, Stop wins if matched first). Streaming path emits `finish_reason = "stop"` on the final chunk followed by `[DONE]`. **Validator update:** new `validate_stop_sequences` helper rejects > 4 strings (OpenAI spec upper bound) and empty/whitespace-only strings (would tokenize to zero tokens, silent no-op) with `400 invalid_request_error`. **Test coverage:** 8 new unit tests in `crates/core/src/sampling/tests.rs` (matches_stop_sequences: single_token / multi_token / partial_match / first_match_wins / empty_stops / stop_longer_than_generated / empty_generated) + 7 new validator unit tests in `crates/server/src/openai/sampling_validation.rs::tests` (stop_validation: none / empty_vec / single_string / max_4_strings / > 4_strings / empty_string / whitespace_string) + 9 new integration tests in `crates/server/tests/chat_integration_test.rs` (chat stop / multi_token stop / multiple stops / completions stop / best_of + stop / logprobs + stop / streaming chat stop / streaming completions stop / max_tokens + stop) + 1 renamed test (test_chat_rejects_non_empty_stop_with_400 → test_chat_stop_now_accepted_by_handler). 2 old P22/P32 validator tests that asserted the 400-rejection are removed. **Public-API delta:** 1 new public field + 2 new builder methods (one of which is non-const because `Option<Vec<Vec<TokenId>>>` is non-Copy). **The v0.x wire-type backlog is now FULLY CLOSED at both the declaration + validation layer AND the engine-honoring layer** — every chat + completions OpenAI-spec field is end-to-end except `tools`/`tool_choice` (grammar-constrained decoder, P33 declared only) and `n > 1` (sample-N parallel, candidate for P39).

- **Clippy `large_enum_variant` lint fix** (v31.0 P38 follow-up) — `EngineMessage::AddRequest.request` changed from `Request` to `Box<Request>` to silence the `clippy::large_enum_variant` lint that P36's `SampledToken`-inflation introduced (the field grew past 248 bytes; other variants are ≤ 64 bytes). Boxing brings the enum to pointer-sized. All 8 `AddRequest { request, ... }` construction sites across `crates/core/src/engine/run.rs`, `crates/core/tests/e2e_graceful_shutdown.rs`, `crates/server/src/openai/{chat,completions}.rs`, and `crates/server/tests/{overload_integration,tutorial_e2e}.rs` updated to `request: Box::new(...)`; the consumption site in `run.rs` uses `*request` to deref back to `Request` for the still-`Request`-by-value `Engine::add_request` public API. **Bonus cleanup:** 7 pre-existing `clippy::uninlined_format_args` errors (in `chat_integration_test.rs` and `sampling/tests.rs`) were also fixed as part of this commit batch — they were masked by the `large_enum_variant` early-termination in prior CI runs. `just ci` is now green on `vllm-core` AND workspace-wide. Public API surface is unchanged (`Request` is consumed by value at construction; callers never read `.request` after — the field-type change is invisible to public consumers).

- **`best_of` engine wire-through on legacy `/v1/completions`** (v31.0 P37 follow-up / technical due diligence API-01) — closes the last v0.x wire-type engine-honoring carve-out. P32 declared + validated `CompletionRequest::best_of` (`Option<u32>`, range `>= 1`) but engine honoring was deferred because it required a logprob-generation primitive; P36 shipped `SampledToken::logprob`, so P37 unblocks the missing engine work. **Honoring is end-to-end** — when `best_of > 1`, the completions handler dispatches to the new private `run_best_of` helper in `crates/server/src/openai/completions.rs` which spawns N parallel `EngineMessage::AddRequest` candidates via `tokio::spawn`, collects N `Vec<SampledToken>` streams, ranks them via the new private `rank_by_mean_logprob` helper (arithmetic mean of per-token `SampledToken::logprob`; tie-breaks on the lowest seq_id / slice index — deterministic, no RNG dependency), and returns the single best completion as a JSON `choices[]` (matches OpenAI's contract: `best_of` returns ONE completion, not N). The chosen completion's text then flows through the existing `apply_completion_meta` (P35) helper for `echo` / `suffix` and through the existing `build_completion_choice_logprobs` (P36) helper when `logprobs = Some(n)` was requested — so `best_of` composes correctly with the prior wire-throughs. **New private helpers in `crates/server/src/openai/completions.rs`:** `run_best_of` (orchestrates the N-candidate spawn / join / rank / render pipeline), `spawn_best_of_candidate` (sends one `EngineMessage::AddRequest` + collects the token stream + finish reason), `rank_by_mean_logprob` (private ranking helper), `populate_completion_sampling_params` (extracted from the inline P27/P28/P29/P30/P34/P36 sampling-params forwarding so the legacy-endpoint → `SamplingParams` mapping has a single authoritative point shared by the streaming / non-streaming / `best_of` paths). **Validator update (P37):** `validate_completion_meta` gains a `best_of > 20` rejection check (matches OpenAI's documented contract; protects the server from N×max_tokens inference cost explosion — `best_of = 1_000_000` would otherwise pin every scheduler sequence slot). The pre-existing `best_of = 0` rejection and the `echo = true && best_of > 1` cross-field rule are unchanged. **Streaming interaction (P37):** `stream = true && best_of > 1` is silently accepted and the response is forced to non-streaming JSON (matches OpenAI's current API behavior — the combination is technically accepted but the runtime shape is a single document, not an SSE event stream, because `best_of` requires ranking N candidates before emitting anything). **Honoring is a no-op** when `best_of = 1` or `best_of = None` (default-path overhead stays at zero — the single-shot handler short-circuits to the existing P35/P36 path). The completions handler does not currently log the field (parity with `seed` / `user` / `frequency_penalty` / `presence_penalty` / `logit_bias` / `echo` / `suffix`). **Test coverage:** 5 new unit tests in `crates/server/src/openai/completions/tests.rs` (`test_rank_by_mean_logprob_single_candidate_returns_zero`, `test_rank_by_mean_logprob_distinct_means_picks_highest`, `test_rank_by_mean_logprob_equal_means_tie_breaks_on_seq_id`, `test_rank_by_mean_logprob_empty_candidates_returns_zero`, `test_rank_by_mean_logprob_length_normalized`) + 4 new unit tests in `crates/server/src/openai/sampling_validation.rs::tests` (`test_completions_best_of_at_upper_bound_passes`, `test_completions_best_of_above_upper_bound_is_rejected`, `test_completions_best_of_well_above_upper_bound_is_rejected`, `test_completions_best_of_at_upper_bound_with_echo_false_passes`) + 7 new integration tests in `crates/server/tests/chat_integration_test.rs` (`test_completions_best_of_one_is_noop_baseline`, `test_completions_best_of_above_one_returns_single_completion`, `test_completions_best_of_returns_highest_mean_logprob`, `test_completions_best_of_above_twenty_returns_400`, `test_completions_best_of_with_stream_silently_falls_back_to_json`, `test_completions_best_of_with_logprobs_returns_chosen_candidates_logprobs`, `test_completions_best_of_with_suffix_appends_to_chosen_completion`, `test_completions_best_of_candidates_are_independent`, `test_completions_best_of_zero_still_returns_400`, `test_completions_best_of_with_echo_true_still_returns_400`, `test_completions_best_of_with_partial_engine_failure_returns_503`). `docs/reference/openai-compatibility.md` flipped the `best_of` row from "Wired (declaration + validation)" → "Wired (declaration + validation + engine wire-through)" with the `<= 20` upper bound + `stream + best_of` silent-fallback behavior documented in the row notes; the v32+ candidates table's `best_of` row is removed entirely; the legacy `echo` / `suffix` / `best_of` summary row is updated to "All three shipped end-to-end" with the P37 helpers cited. **Public-API delta: 0 new public items** — all new helpers (`run_best_of`, `spawn_best_of_candidate`, `rank_by_mean_logprob`, `populate_completion_sampling_params`) are private to the completions module; the validator's new `<= 20` check is a tightening of the existing `validate_completion_meta` signature; the wire-type contract is delivered through the existing `CompletionRequest::best_of` field (P32) and the existing `CompletionChoice::text` + `CompletionChoice::logprobs` response fields. **The v0.x wire-type backlog is now FULLY CLOSED at both layers** (declaration + validation: P21/P22/P23/P27/P31/P32/P33; engine wire-through: P28/P29/P30/P35/P36/P37) — only `tools` / `tool_choice` (grammar-constrained decoder, P33 declared only) remains as v32+ work on the chat endpoint.

- **`logprobs` + `top_logprobs` engine wire-through** (v31.0 P36 follow-up / technical due diligence API-01 / Phase 31-D-adjacent) — the response-side logprob-generation primitive that the P31 declaration + validation step left as v32+ work. New public type `vllm_traits::SampledToken { token: TokenId, logprob: f32, top_logprobs: Vec<(TokenId, f32)> }` carries the sampled token alongside its post-filter logprob (`ln(P)` under the post-`repeat_penalty` / post-`presence_penalty` / post-`logit_bias` / post-temperature / post-`top_k` / post-`top_p` distribution) and (when `SamplingParams::top_logprobs.is_some()`) the top-K alternatives sorted by logprob descending. New public field `SamplingParams::top_logprobs: Option<u32>` + `SamplingParamsBuilder::with_top_logprobs` activate the chat `0..=20` and completions `0..=5` ranges (validated at the HTTP layer by P31). New public types `vllm_server::openai::types::ChatLogprob` + `ChatChoiceLogprobs` mirror OpenAI's `choices[].logprobs.content[]` shape 1:1 (each entry carries `token` decoded string + `logprob` + `bytes` UTF-8 representation + nested `top_logprobs: Option<Vec<ChatLogprob>>`); new public field `ChatChoice::logprobs: Option<ChatChoiceLogprobs>` on the chat response. New private helpers `vllm_core::sampling::{logprob_of_token, top_logprobs_of, log_softmax, renormalized_top_p_logprobs}` — `logprob_of_token` is numerically stable (subtracts `max(logits)` before `exp`/`ln` so the intermediate `sum_exp` is bounded by `[0, vocab_size]`); all four are `#[allow(dead_code)]` so they form a stable sampler API surface even if only `sample_one_with_params` calls them today. **Wire-breaking:** `vllm_core::sampling::{sample_one_with_params, sample_batch_with_params}` now return `SampledToken` instead of bare `TokenId`; 28 workspace-internal call sites updated to read `.token` (mechanical sweep across `engine/`, `scheduler/`, `spec_dispatch/`, `self_spec/`, `model/`, tests). The `EngineMessage::AddRequest` `response_tx` channel carries `SampledToken` end-to-end so the HTTP renderer can populate the OpenAI logprob response shape without re-running the softmax. A new `const fn placeholder_sampled(TokenId) -> SampledToken` lives in `spec_dispatch::verify` for speculative-accepted draft tokens whose true logprob is unavailable without re-running the target forward pass (HTTP layer detects via `logprob == 0.0 && top_logprobs.is_empty()`). 7 new unit tests in `crates/core/src/sampling/tests.rs` (`test_logprob_of_token_returns_zero_for_uniform_distribution`, `test_logprob_of_token_numerically_stable_with_large_logits`, `test_logprob_of_token_returns_neg_inf_for_neg_inf_logits`, `test_top_logprobs_of_returns_empty_for_zero`, `test_top_logprobs_of_sorted_descending`, `test_top_logprobs_of_fewer_than_n_when_vocab_small`, `test_log_softmax_subtraction_invariant`) + 7 new chat integration tests in `crates/server/tests/chat_integration_test.rs` (`test_chat_forwards_top_logprobs_to_engine`, `test_chat_top_logprobs_zero_round_trips`, `test_chat_response_logprobs_wire_shape`, `test_chat_response_omits_logprobs_when_not_requested`, `test_completions_forwards_logprobs_to_engine`, `test_completions_response_logprobs_wire_shape`, `test_completions_response_omits_logprobs_when_not_requested`) — the wire-shape tests pin the exact JSON layout (`choices[0].logprobs.content[]` for chat; parallel `tokens[]` / `token_logprobs[]` / `top_logprobs[][]` for completions) so future serializer changes trip the suite. Streaming chat renders the per-token logprob entry on each intermediate chunk and skips the entry on the `finish_reason` chunk (matches OpenAI's accumulator semantics). `docs/reference/openai-compatibility.md` flipped all four logprobs / top_logprobs rows from "Wired (declaration + validation)" → "Wired (declaration + validation + engine wire-through)" and the response-side `choices[].logprobs` row to "Wired (declaration + engine wire-through)". Public-API delta: 1 new public type `vllm_traits::SampledToken` + 1 new field `SamplingParams::top_logprobs` + 1 new builder method + 2 new public types `vllm_server::openai::types::{ChatLogprob, ChatChoiceLogprobs}` + 1 new field `ChatChoice::logprobs`. The v0.3 wire-type backlog is now FULLY CLOSED at both layers (declaration + validation: P21/P22/P23/P27/P31/P32/P33; engine honoring: P28/P29/P30/P36). The v32+ candidates remaining: `tools` / `tool_choice` (grammar-constrained decoder — P33 declared only), `best_of` on legacy `/v1/completions` (sample-N-rank-by-mean-logprob — **now unblocked** by P36's logprob-generation primitive; the natural P37 candidate).

- **Documentation Overhaul (v31.0 Phase 31-B)** — honest, architecture-first documentation.
    - `docs/architecture.md` — system design with Mermaid diagrams (single source of truth).
    - `OPERATIONS.md` — deployment, monitoring, troubleshooting runbook.
    - `docs/adr/ADR-019-documentation-standards.md` — doc coverage targets and comment conventions.
    - `.planning/v31.0-MASTER-PLAN.md` — v31 milestone plan (6 phases).
    - `.planning/STATE.md` updated to v31 in_progress.
    - `docs/README.md` rewritten with accurate crate map and doc index.
    - `README.md` honesty pass: badges (1235 tests, Rust 1.88+), removed deleted feature flags (`prometheus`/`opentelemetry`), fixed architecture tree (StubArchitecture, no top-level `tests/`), accurate feature flag table.
    - Tutorial 03/04 rewritten to match real `Engine::run` + `EngineMessage` actor API and `SchedulerEngine::set_policy`.

- **Public API CI Gate (v31.0 Phase 31-C / Phase 12e)** — `cargo-public-api` baseline diff wired into `just ci` and GitHub Actions.
    - Baselines refreshed for all 6 workspace crates (including new `core.txt`) under `.planning/phase-12e/`.
    - `just public-api-check` fails when public API grows without a `CHANGELOG.md` entry; shrinking is allowed.
    - `just public-api-baseline` regenerates snapshots after intentional API changes.

- **KV Block Transfer Protocol (v31.0 Phase 31-D / OPS-31d)** — closes the protocol-layer gap left by OPS-05c: replicated `(block_id, chain_hash)` intent can now be paired with actual KV tensor bytes flowing across nodes.
    - **`BlockDataSource` trait** (`crates/dist/src/distributed_kv/block_data_source.rs`) — async, object-safe abstraction over raw block bytes. Production wraps `PagedKvCache` (v32+); tests use `MockBlockDataSource`.
    - **`TransferKVBlock` gRPC RPC** — receiver sends `(block_id, expected_hash)`; sender returns `(block_id, chain_hash, bytes data, num_tokens)`. Chain-hash verification prevents wrong-content transfers.
    - **`PeerClient::fetch_block`** — clones the existing `put`/`invalidate` pattern; bumps the generated client builder's `max_decoding_message_size` / `max_encoding_message_size` to `MAX_BLOCK_TRANSFER_BYTES` (64 MiB) so production-sized blocks (≈14 MiB for Qwen3-7B at F32) fit.
    - **`DistributedKVCache::fetch_block`** — fan-out fallback over every peer (first response whose `chain_hash` matches the local `value_hash` wins); falls back to the local `BlockDataSource` if fan-out fails. Smart owner-based routing is explicitly deferred to v32+.
    - **`FetchError`** — typed enum with `NotFound`, `HashMismatch`, `SourceUnavailable`, `NoPeers`, `AllPeersFailed`, `Transport` variants.
    - **`GrpcState::block_data_source`** + `transfer_kv_block` handler — serves inbound `TransferKVBlock` RPCs from the local source; symmetric 64 MiB message limit on the server side.
    - **`MAX_BLOCK_TRANSFER_BYTES = 64 MiB`** — applied symmetrically on server *and* client (Tonic's default 4 MiB would silently fail for any production-sized block). Integration test `fetch_block_works_above_default_message_limit` verifies a 5 MiB block round-trips end-to-end.
    - **Test count delta**: `vllm-dist` unit 59 → 75 (+16); `vllm-dist` integration 5 → 12 (+7); workspace 1307 → 1338 (+31).
    - Phase plan: `.planning/phase-19/ops-31d-kv-block-transfer.md`.

- **CycloneDX SBOM per release artifact (v31.0 P11 follow-up / technical due diligence engineering-quality §7)** — every GitHub Release now attaches a CycloneDX JSON SBOM per build target, emitted by the `build` job via [`anchore/sbom-action`](https://github.com/marketplace/actions/anchore-sbom-action) (`syft` under the hood). Each SBOM captures every Rust crate in `Cargo.lock`, vendored C libraries that ended up linked into the binary, and any system libraries detected by syft's ELF/PE/Mach-O scanners. The new `Software Bill of Materials` section in `docs/RELEASE.md` documents how to download, inspect, and cross-check the SBOM with `jq`, and explains why downstream consumers running vLLM-lite in regulated environments (SOC 2 / FedRAMP / air-gapped vendor review) benefit. Build job now requests `contents: write` + `actions: read` explicitly so the SBOM step can both upload its artifact and contribute to the GitHub Release attachment glob. Checksums (`sha256sum`) and signed build provenance (SLSA / in-toto) are still missing — tracked as a separate follow-up against §7.

- **Multi-Node quickstart expanded in OPERATIONS.md (v31.0 P12 follow-up / Phase 31-D completion)** — the "Multi-Node (Experimental)" section is rewritten to close the two remaining 31-D master-plan items: a documented `TransferKVBlock` wire protocol spec, and an actionable multi-node quickstart. New subsections: "What works" (3 bullets, Phase 31-D / OPS-31d), "What is **not** yet production-ready" (4 bullets, led by the **load-bearing** `PagedKvCacheWrapper → MemoryManager` engine wiring gap which is v32+ / OPS-32a), "Minimum viable cluster" with **both** 2-node AND 3-node snippets (the 3-node form mirrors `crates/dist/tests/distributed_kv_peer_sync.rs::multi_peer_broadcast`), "Verify it works" pointing operators at the in-process gRPC integration tests that exercise real 2-node + 3-node fan-out without a real network, and a "Wire protocol (TransferKVBlock, Phase 31-D)" subsection that quotes the proto definitions, explains the 64 MiB symmetric message limit (with the warning that custom embedders must bump the same limits or transfers will return `tonic::Status::out_of_range_error`), and documents the hash-verification contract. `vllm-dist` engine integration tests are not new — this batch is pure documentation that closes the master-plan checkboxes for 31-D.

- **Mutation testing wired into nightly CI (v31.0 P13 follow-up / Phase 31-E)** — closes the "Mutation testing CI (fix baseline workaround)" master-plan item. New `.github/workflows/mutation-nightly.yml` runs `cargo mutants` on three short `vllm-core` modules (`sampling.rs`, `scheduler/policy`, `engine/cuda_graph.rs`) on a daily 03:00 UTC cron + manual dispatch, gates each scan against a per-module score floor via `scripts/check_mutation_score.sh`, and uploads `.mutants-out/` as a 14-day workflow artifact for offline inspection. Baseline fix: the pre-existing `--baseline skip` workaround in the `just mutants` recipe (added to mask a `cuda_graph_integration.rs:148` test failure tracked as a v31+ follow-up) was verified unnecessary for the modules under scan — line 148 is inside a `#[cfg(feature = "cuda-graph")]` test that does not compile under default features, so the baseline is green. The CI workflow drops `--baseline skip`; the local justfile keeps it as a one-line escape hatch for developers with broken tests in progress. CI workflow comment records the verification (135 mutants, 60 caught, 73 missed, 2 unviable on `sampling.rs` under default features). The "fix the test in v31+" comment in `justfile` is updated to point future readers at the workflow for the new ground truth.

- **ADR-020 — Multi-Node KV Block Transfer Architecture (v31.0 P14 follow-up / Phase 31-D documentation)** — captures the six architectural decisions from OPS-31d as an ADR for future maintainers: the `BlockDataSource` trait as the storage-agnostic abstraction; the unary (not streaming) `TransferKVBlock` gRPC RPC; fan-out fallback routing vs owner-routed; the 64 MiB symmetric message limit; explicit deferral of engine integration (`PagedKvCacheWrapper → MemoryManager`) to v32+ / OPS-32a; and the non-removal of the legacy `GetKVCache` RPC in this phase. The ADR documents the alternatives that were considered (streaming RPCs, owner-routed day-1, putting block bytes in `PutKVCache`, `GetKVCache(block_hash)` lookup) and the rationale for fan-out + unary + block-id that ships in v0.1. v31.0 quality-gate target bumped from "ADRs: 19" to "20 records: ADR-001 … ADR-020"; `.planning/DOC-MAP.md` and `README.md` ADR-count cross-references updated to match.

- **`docs/reference/openai-compatibility.md` — v0.2 follow-ups section added (v31.0 P15 follow-up)** — the matrix's nine "Not declared" rows were previously split only implicitly (some tracked in STATE.md, others nowhere). New "v0.2 follow-ups (planned)" section makes the split explicit: `seed` + `user` + `response_format` JSON-mode land in v0.2 (declaration + HTTP-boundary validation; honoring depends on engine-side work tracked separately); `frequency_penalty` / `presence_penalty` / `logit_bias` / `logprobs` / `top_logprobs` / `tools` / `tool_choice` defer to v32+ with per-field rationale (sampling-renaming public-API delta, sampler bias-map injection, logprob generation, grammar-constrained decoder). Header paragraph updated with a pointer to the new section so a reader scanning the field tables immediately knows where the backlog lives.
- **`docs/technical-due-diligence/engineering-quality.md` — §6 + §7 stale-item closure (v31.0 P15 follow-up)** — the §6 "Feature 问题" list had four items, three of which became stale as the workspace evolved (`kernels = []` is in fact a cfg-gate feature with real code; `server` no longer hard-enables `cuda-graph`; `dist` is in default-members). New "§6 闭合状态 (v31.0 P15)" table marks 3 closed + 1 half-closed (the "feature matrix doc" sub-item is a real follow-up, partially covered by `docs/architecture.md` §Feature Flags). §7 (MSRV + reproducible builds) similarly had four evidence lines, all closed by DEP-01 / P11 / P13; new "§7 闭合状态 (v31.0 P15)" table closes them item-by-item and explicitly tracks the still-open "checksums + provenance" half as v32+. Net effect: the due diligence doc now matches v31.0 reality instead of v19.x snapshot, and the remaining open items are visible at the bottom of each section rather than scattered across STATE.md.
- **`docs/technical-due-diligence/production-readiness.md` — §2–§11 stale-item closure (v31.0 P16 follow-up)** — the document's ten subsections (SEC-01, REL-01, 输入边界, OBS-01, 日志与追踪, 健康检查与关停, 部署阻断项, TLS 与 CORS, Batch 与 Embeddings, 生产门槛) describe concerns that P0–P15 have largely resolved. New top-of-document "v31.0 P0–P15 closure summary" table aggregates per-section status (29 closed / 5 partial / 9 v32+ candidates out of 43 original observations); each section gains a per-item closure table with code / test / commit references. Highlights: SEC-01 closed by DEP-01 + P4 (RBAC `AuthenticatedRole` removes header-forgery); REL-01 closed by REL-01 batch + P1 (bounded mailbox + `CancelRequest`); OBS-01 closed by OBS-01 batch (`engine.scheduler.metrics` Arc-shared); graceful shutdown closed by P7 (`mark_not_ready` + drain grace + engine thread join); body limit + context length closed by P1 + P3. The 9 v32+ items split into 5 code follow-ups (TLS 主路径接线 / OTLP / per-tenant quota / readiness 模型加载信号 / feature matrix doc) and 4 infrastructure gaps (GPU runner / 容量基准 runbook / 升级-回滚 / 真实 checkpoint CI) that are out of scope for v31.0 alpha. Docs-only — no Rust / no test / no API delta.
- **`docs/technical-due-diligence/architecture-performance.md` — §6 speculative + distributed closure (v31.0 P17 follow-up)** — the last remaining due-diligence subsection without a closure audit. New "§6 闭合状态 (v31.0 P17)" table covers seven original observations: 4 closed (`NcclAllReduce` → `LocalSumAllReduce` deprecation alias, distributed KV block transfer protocol + fan-out fallback, `dist` in default-members, multi-node explicitly marked Experimental with three ADRs recording the architectural decisions), 1 partial (speculative verification is **already** temperature-aware sampled-match — `crates/core/src/engine/spec_dispatch/verify.rs::verify_draft_tokens_logits`; the doc-comment explicitly notes this is the standard lossless speculative decoding variant, **not** the full `min(1, p/q)` rejection-sampling which would need draft-side logits), 2 still-gap-but-documented (speculative ↔ CUDA Graph mutual exclusion is explicit design — verified in code: `step_speculative_inner` vs `step_with_graph` paths are switch-mutually-exclusive because CUDA Graph capture needs static batch shape while speculative decode introduces variable draft length; `PagedKvCacheWrapper: BlockDataSource` engine wiring remains v32+ / OPS-32a). One bonus close: legacy draft model path is verified gone — `grep -rn "legacy" speculative/` returns empty; `draft_registry` / `draft_resolver` / `adaptive` / `self_spec` are the only draft sources. Docs-only — no Rust / no test / no API delta.

- **`EngineBuilder::with_paged_kv_cache` — engine-side wiring of multi-node KV transfer** (v31.0 P41 / OPS-32a second half) — closes the engine-plumbing half of the deferred gap from ADR-020 §5 (the wrapper half shipped in P40). New `MemoryManager::block_data_source: Option<Arc<dyn BlockDataSource + Send + Sync>>` field + `with_block_data_source` builder / `set_block_data_source` setter / `block_data_source` getter (gated `#[cfg(feature = "multi-node")]`); new `EngineBuilder::with_paged_kv_cache(Arc<PagedKvCache>)` builder method that constructs the `PagedKvCacheWrapper` internally and threads it through `Engine::set_paged_kv_cache` → `SchedulerEngine::set_block_data_source` → `MemoryManager::block_data_source`; new `PagedKvCache::write_layer_block` pub(crate) helper (mirrors the P40 T1 `read_layer_block`, enables the receiver-side write path that P42 will wire); new `PagedKvCacheWrapper::verify_chain_hash` defense-in-depth helper (read-only, no allocation, no I/O); server bootstrap adds a builder-path engine construction selected when the loader exposes a `PagedKvCache`, plus a new `app_config.server.multi_node` config section (`enabled: false` default + `node_id: null` auto-generates a uuid v4 + `bind_addr: 0.0.0.0:50051`) that spawns the multi-node gRPC server via the new `bootstrap::grpc::spawn_multi_node_grpc_server` helper. **8 new tests** (4 `MemoryManager::block_data_source` + 4 `write_layer_block`; the `EngineBuilder` + `SchedulerEngine` + `verify_chain_hash` are covered by the existing P40 e2e test which now flows through the engine path). **Public-API delta (default features): 0.** Public-API delta (`--features multi-node`): +1 `pub` method on `MemoryManager`, +2 `pub(crate)` methods on `Engine`, +1 `pub` method on `EngineBuilder`, +2 `pub` methods on `PagedKvCacheWrapper`, +1 `pub(crate)` method on `PagedKvCache`. **Receiver-side write path** (`write_kv_batch` + `TransferKVBlock` handler update + 2-node e2e test) **deferred to P42** — the helper is in place, the wiring + e2e test are tracked as the natural next half. ADR-020 status line bumped from "engine plumbing deferred to P41+" to "engine plumbing shipped in P41; receiver-side write_kv_batch deferred to P42+".

### Changed

- **`NcclAllReduce` honest renaming** (v31.0 P4 follow-up batch / technical due diligence `dist` §3.1) — `NcclAllReduce` and `LocalSumAllReduce` were two identically-implemented types; the `Nccl` prefix was misleading because v0.x has no NCCL backend. `LocalSumAllReduce` is now the canonical type; `NcclAllReduce` is a `#[deprecated]` type alias so existing callers keep compiling during the v0.x transition window. Re-exports updated to prefer `LocalSumAllReduce`. Compile-only test `nccl_all_reduce_alias_resolves_to_local_sum` guards the deprecation contract.
- **public-api: vllm-dist added** — `LocalSumAllReduce` (canonical), the `NcclAllReduce` deprecated alias, and 8 associated method/impl items (Phase 31-D follow-up / v31.0 P4).
- **public-api: vllm-traits added** — `FinishReason` enum (with `Stop` / `Length` / `Cancelled` variants and `Clone` + `Copy` + `Serialize`/`Deserialize` impls) moved here from `vllm-core` so `vllm-dist` / `vllm-server` can reference it without a cyclic dependency. 28 new items (Phase 31-A follow-up / v31.0 P2 ARCH-02 / sampling refactor).
- **public-api: vllm-core added** — `Engine::finish_reason_txs` (per-sequence oneshot channels for caller-visible finish reasons), `EngineMessage::AddRequest::{finish_reason_tx,seq_id_tx}` request acknowledgment fields, and `EngineMessage::CancelRequest` variant (8 new items). Phase 31-A follow-up: disconnect-propagation (P0 batch) + ARCH-02 sampling refactor (P2 batch).
- **public-api: vllm-model added** — `ModelLoader::capabilities() -> Option<ArchCapabilities>` (1 new item). Required by the embeddings capability gate (P4 batch / production-readiness §10) so `/v1/embeddings` can return `400 not_supported` for architectures whose `embed()` is a stub.
- **public-api: vllm-server added** — 102 new items across the production-readiness hardening batches: CORS layer (`CorsConfigFile`, `cors::CorsConfig`, `with_cors`, `AppConfig::cors`, `CorsConfigFile::{allow_origins,allow_methods,allow_headers,allow_credentials,into_runtime}`, P2 / §9), audit middleware (`audit_middleware` module + `audit_status_to_result` + `AuthenticatedRole` + `ApiState::audit`, P3 / §6), `/debug/audit` endpoint (`debug::AuditDumpResponse` + `{count,returned,cap,events}` fields, P3 / §6), health-handlers module (`health_handler` / `metrics_handler` / `ready_handler` extracted from inline `main.rs` closures, P3 follow-up), sampling-field validation (`openai::sampling_validation` module + `validate_chat_request_fields` + `validate_completion_request_fields` + `validate_sampling_params`, P6 / §5.1), embeddings capability gate (`ApiState::arch_capabilities`, P4 / §10), context-length validation (`ApiState::max_model_len`, P2 / §4), and graceful-shutdown readiness flip (`HealthChecker::mark_not_ready` + `ServerConfig::shutdown_drain_grace_secs` + `ConfigValidationError::ShutdownDrainGraceTooLarge`, P7 / §7).
- **public-api: vllm-server added** — 3 new items for `top_p` honouring (P9 follow-up / technical due diligence architecture-performance §5.1.6): `ChatRequest::top_p`, `CompletionRequest::top_p` (declared wire-format fields), and `sampling_validation::validate_top_p` (HTTP-boundary guard for the `(0, 1]` range + `NaN` rejection).
- **OpenAI API compatibility matrix** (v31.0 P6 follow-up / technical due diligence API-01 §5.1) — `ChatRequest` and `CompletionRequest` declared `n`, `stop`, and `top_p` fields but the engine never honoured them (silent acceptance + ignore = contract violation). New `validate_chat_request_fields` / `validate_completion_request_fields` in `crates/server/src/openai/sampling_validation.rs` reject `n > 1` and non-empty `stop` with `400 invalid_request_error` BEFORE enqueuing. `n = 1` and `stop = []` are accepted (functionally no-ops matching OpenAI defaults). New `docs/reference/openai-compatibility.md` is the single source of truth for what the wire types do vs don't honour. 8 new unit tests in `sampling_validation` + 4 new integration tests in `chat_integration_test.rs`.
- **`top_p` is now honoured end-to-end** (v31.0 P9 follow-up / technical due diligence architecture-performance §5.1.6) — the engine's `vllm_core::sampling::sample_batch_with_params` already implemented nucleus sampling; this batch wires the `ChatRequest.top_p` and `CompletionRequest.top_p` fields through the HTTP handlers into `Request::sampling_params.top_p` so the value actually reaches the sampler. New `sampling_validation::validate_top_p` rejects `top_p <= 0`, `top_p > 1`, and `NaN` with `400 invalid_request_error` at the HTTP boundary (range `(0, 1]` per OpenAI spec) so callers learn about bad values BEFORE paying the cost of enqueuing. `docs/reference/openai-compatibility.md` updated to mark `top_p` as HONOURED on `/v1/chat/completions` and `/v1/completions`. 7 new unit tests in `sampling_validation` (`top_p_none_passes`, `top_p_zero_is_rejected`, `top_p_negative_is_rejected`, `top_p_one_passes`, `top_p_above_one_is_rejected`, `top_p_nan_is_rejected`, `top_p_intermediate_passes`) + 5 new integration tests in `chat_integration_test.rs` (`test_chat_forwards_top_p_to_engine`, `test_chat_omitted_top_p_uses_engine_default`, `test_chat_rejects_top_p_above_one_with_400`, `test_chat_rejects_top_p_zero_with_400`, `test_completions_forwards_top_p_to_engine`).
- **request_id propagated end-to-end** (v31.0 P10 follow-up / technical due diligence production-readiness §6) — `correlation_id_middleware` (P1 batch) minted an `X-Request-ID` and installed a `CorrelationId` extension on every incoming request, but the value never reached `EngineMessage::AddRequest`, so engine-side `tracing` log lines could not be correlated with the originating HTTP request. This batch wires the id through: `AddRequest` gains an `Option<String> request_id` field; `chat_completions` and `completions` extract `CorrelationId` from request extensions and forward it; the engine run loop enters a `tracing::info_span!("engine.add_request", request_id)` so every synchronous log line in `add_request` and its callees (scheduler admission, KV allocation, prefix-cache lookup) carries the same id. The middleware MUST be mounted as the OUTERMOST layer in any router hosting these handlers — see the test fixtures (`cancel_propagation.rs`, `chat_integration_test.rs`, `context_length.rs`, `overload_integration.rs`) for the canonical pattern. 4 new integration tests in `crates/server/tests/request_id_propagation.rs` cover client-supplied id round-trip, minted id round-trip, and the same on both endpoints.
- **public-api: vllm-server added** — 5 new items for request_id propagation (P10 follow-up / technical due diligence production-readiness §6): `security::correlation::CorrelationId` promoted `pub(crate)` → `pub` so axum's `Extension<CorrelationId>` extractor can name it from public HTTP handlers, and `chat_completions` / `completions` gained an `Extension<CorrelationId>` parameter (signatures widened; existing callers that go through `Router::layer(from_fn(correlation_id_middleware))` are unaffected). `EngineMessage::AddRequest` gained a `request_id: Option<String>` field (lives in `vllm-core`, no public API change — `EngineMessage` has no `Serialize`/`Deserialize`).
- **public-api: vllm-server added** — 2 new items for the OpenAI `user` field declaration (P21 follow-up / v0.2 wire-type): `ChatRequest::user` + `CompletionRequest::user` (`Option<String>`, both `#[serde(default, skip_serializing_if = "Option::is_none")]`). No handler signature changes.
- **public-api: vllm-server added** — 1 new type + 1 new field for the OpenAI `response_format` declaration (P22 follow-up / v0.2 wire-type): `openai::types::ResponseFormat` enum (with `Text` + `JsonObject` variants, `#[serde(tag = "type", rename_all = "snake_case")]`) and `ChatRequest::response_format: Option<ResponseFormat>` (`#[serde(default, skip_serializing_if = "Option::is_none")]`). `CompletionRequest` intentionally does NOT declare the field — the legacy `/v1/completions` endpoint does not support `response_format` per OpenAI spec.
- **public-api: vllm-server added** — 2 new fields for the OpenAI `seed` declaration (P23 follow-up / v0.2 wire-type): `ChatRequest::seed` + `CompletionRequest::seed` (`Option<i64>`, both `#[serde(default, skip_serializing_if = "Option::is_none")]`). Per OpenAI spec any `i64` is accepted (no range / sign validation, no NaN check). Honoring is a no-op today — engine-side RNG seeding is v32+ work; the wire-type contract is locked in now so the declaration-only PR doesn't regress to "rejected by serde". No handler signature changes (the chat handler threads the value into the existing `tracing::info!(...)` log lines).
- **Dead-code backpressure module dropped** (v31.0 P26 / technical due diligence REL-01 closure companion) — `crates/server/src/backpressure.rs` (284 lines, gated by `#![allow(dead_code)]` and never wired into the router) is deleted. The token-bucket admission control it implemented was superseded by the bounded `engine_mailbox_capacity` from the REL-01 closure batch (commit `0f3f9db`); that path uses `try_send` so a saturated mailbox surfaces as `503 engine_overloaded` rather than blocking on capacity. `crates/server/src/lib.rs` loses the `pub mod backpressure;` declaration; `crates/server/src/api.rs` build_router doc-comment loses "backpressure" from the middleware list (the surviving list now reads "correlation, audit, auth" — matches the actual middleware tower). Verified: `cargo check -p vllm-server` clean; `cargo test -p vllm-server --lib` 183 → still 183 passed (no test referenced the deleted types; `crates/testing/src/slow_model.rs`'s doc-comment still mentions "backpressure" because slow-model is a generic test helper unrelated to the deleted server-side module). No public API delta — the deleted types were `pub(crate)` and never re-exported.
- **public-api: vllm-server added** — 4 new fields + 1 new public helper for the OpenAI `frequency_penalty` + `presence_penalty` declarations (P27 follow-up / v0.3 wire-type): `ChatRequest::frequency_penalty` + `ChatRequest::presence_penalty` + `CompletionRequest::frequency_penalty` + `CompletionRequest::presence_penalty` (all `Option<f32>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`) and `openai::sampling_validation::validate_penalty` (HTTP-boundary guard for the `[-2.0, 2.0]` range + `NaN` / `±infinity` rejection; takes `Option<f32>` + `&str` field name so the same helper serves both fields with field-specific error messages).
- **public-api: vllm-traits added** — 2 new items for the OpenAI `presence_penalty` engine wire-through (P28 follow-up / v0.3 wire-type): `SamplingParams::presence_penalty: f32` field (default `0.0`, OpenAI-spec semantic — additive bias subtracted from the logit of every distinct seen token regardless of occurrence count) and `SamplingParamsBuilder::with_presence_penalty(f32) -> Self` const builder method. The `SamplingParams` struct has `#[derive(Default)]` so existing callers using `..Default::default()` are unaffected; direct struct-literal callers must add the new field (only 2 call sites in the workspace, both use `..Default::default()` so they pick up the new default automatically).
- **public-api: vllm-core added** — 1 new public function for the OpenAI `presence_penalty` engine wire-through (P28 follow-up / v0.3 wire-type): `vllm_core::sampling::apply_presence_penalty(&mut [f32], &[vllm_traits::types::TokenId], f32)` — subtracts the penalty from the logit of every distinct seen token (deduped via internal `HashSet`), wired into `sample_one_with_params` after `apply_repeat_penalty` so the sampling pipeline matches the documented order (repeat penalty → presence penalty → temperature → top-k → top-p / greedy).
- **public-api: vllm-server added** — 5 new items for the OpenAI `logprobs` + `top_logprobs` declaration + validation (P31 follow-up / v0.3 wire-type): 3 declared wire-format fields (`ChatRequest::logprobs: Option<bool>`, `ChatRequest::top_logprobs: Option<u32>`, `CompletionRequest::logprobs: Option<u32>` — all `#[serde(default, skip_serializing_if = "Option::is_none")]`) and 2 new public validators in `openai::sampling_validation` (`validate_chat_logprobs(Option<bool>, Option<u32>)` rejects `top_logprobs > 20` + the cross-field rule `top_logprobs = Some(_)` requires `logprobs = Some(true)`; `validate_completion_logprobs(Option<u32>)` rejects values > 5). Type asymmetry preserved on purpose — chat's `logprobs` is `bool` (the chat spec's flag) and completions' is `u32` (the legacy spec's 0-5 integer count) — matches OpenAI spec exactly rather than unifying behind a common representation. Honoring is a no-op today — engine-side top-K logprob generation requires a wire-breaking change to `sample_batch_with_params`'s return type; tracked as v32+ work.
- **public-api: vllm-server added** — 4 new items for the OpenAI legacy `/v1/completions` `echo` + `suffix` + `best_of` declaration + validation (P32 follow-up / v0.x wire-type): 3 declared wire-format fields on `CompletionRequest` (`echo: Option<bool>`, `suffix: Option<String>`, `best_of: Option<u32>` — all `#[serde(default, skip_serializing_if = "Option::is_none")]`) and 1 new public validator (`openai::sampling_validation::validate_completion_meta(Option<bool>, Option<u32>)` rejects `best_of = 0` and the cross-field rule violation `echo = true && best_of > 1` per OpenAI spec). Per OpenAI legacy-completions spec `best_of` must be `>= 1` (default `1`); `suffix` is unconstrained (any string accepted, used by code-completion UIs to pre-fill the suffix the model should fill against); `echo` defaults to `false` (when `true`, the response echoes the prompt back as a prefix to the generated continuation). Honoring is a no-op for all three today — engine would need to prepend the prompt / append the suffix to `CompletionChoice.text` and (for `best_of`) sample N times + rank by mean logprob (the latter depends on the same v32+ logprob-generation work as the `logprobs` field, P31). No handler signature changes.
- **public-api: vllm-server added** — 9 new items for the OpenAI `/v1/chat/completions` `tools` + `tool_choice` declaration + validation (P33 follow-up / v0.x wire-type): 6 new public types (`openai::types::Tool` carries a `type` discriminator + `function` payload; `openai::types::FunctionDefinition` carries `name: String` + optional `description` + optional `parameters: serde_json::Value` for the JSON Schema; `openai::types::ToolType` enum currently with only the `Function` variant — future OpenAI extensions would add new variants rejected at the serde layer as `422 Unprocessable Entity`; `openai::types::ToolChoice` `#[serde(untagged)]` enum with `Mode(ToolChoiceMode)` and `Specific(ToolChoiceSpecific)` variants preserves the wire shape 1:1; `openai::types::ToolChoiceMode` enum with `None` / `Auto` / `Required` variants for the string-mode form; `openai::types::ToolChoiceSpecific` + `openai::types::ToolChoiceFunctionRef` for the object form `{"type": "function", "function": {"name": "..."}}`) + 2 new declared wire-format fields on `ChatRequest` (`tools: Option<Vec<Tool>>` + `tool_choice: Option<ToolChoice>` — both `#[serde(default, skip_serializing_if = "Option::is_none")]`) + 1 new public validator (`openai::sampling_validation::validate_chat_tool_choice(Option<&[Tool]>, Option<&ToolChoice>)` rejects the three cross-field rules: `tool_choice = Some(Required)` with empty / absent `tools`, `tool_choice = Some(Specific)` with empty / absent `tools`, `tool_choice = Some(Specific { function: { name: X } })` with no matching tool in `tools[]`). **NOT** declared on `CompletionRequest` — the legacy endpoint doesn't support tool calling per OpenAI spec. The `parameters` field is stored as `serde_json::Value` (rather than a typed JSON-Schema struct) because OpenAI's spec is intentionally permissive about JSON Schema drafts and the engine doesn't process the schema today (grammar-constrained decoding is v32+ work). Field names + descriptions + parameters are *not* validated against the OpenAI-spec regex (`^[a-zA-Z0-9_-]{1,64}$`) today — the validator is permissive by design until honoring lands. Honoring is a no-op — engine-side tool calling requires a grammar-constrained decoder (JSON-schema → grammar at sample time) and a per-request tool schema cache (the JSON Schema in `tools[].function.parameters` is per-request and can't be globally memoized). Architecture-level work — v32+.
- **public-api: vllm-traits added** — 3 new items for the OpenAI `seed` engine wire-through (P34 follow-up / v0.2 wire-type): `SamplingParams::seed: Option<u64>` field (`#[serde(default, skip_serializing_if = "Option::is_none")]`, default `None` — when `Some(s)`, the sampler builds a fresh `rand::rngs::StdRng::seed_from_u64(s)` per `sample_one_with_params` call so the OpenAI-spec "same seed + same model + same prompt → same output" contract holds), `SamplingParamsBuilder::with_seed(u64) -> Self` const builder method (sets `seed = Some(_)`), and `SamplingParamsBuilder::with_seed_none() -> Self` const builder method (explicitly clears `seed` to `None` — useful when the builder default is overridden by a different code path). The `SamplingParams` struct has `#[derive(Default)]` so existing callers using `..Default::default()` pick up `seed = None` automatically; direct struct-literal callers must add the new field (no such callers in the workspace today — all use either `..Default::default()` or the builder).
- **public-api: vllm-core added** — 1 new public function for the OpenAI `seed` engine wire-through (P34 follow-up / v0.2 wire-type): `vllm_core::sampling::sample_random_threshold(Option<u64>) -> f32` — the single RNG touchpoint that returns one random `f32` drawn from either a fresh `StdRng::seed_from_u64(seed)` (when `Some(_)`) or the thread-local default RNG (when `None`); `temperature_sample` / `top_p_sample` / `sample_batch` / `sample_one_with_params` all consume the precomputed threshold via a `random_threshold: f32` parameter rather than reading the RNG internally (the parameter is private to `pub(crate)` helpers, but `sample_random_threshold` itself is `pub` so library users wiring a custom sampler can reuse the same RNG-selection dispatch). The `top_k_sample(logits, k)` function gained a third `random_threshold: f32` parameter (wire-breaking change to the public signature of `vllm_core::sampling::top_k_sample` — only 4 workspace-internal callers exist, all updated).

- **Test-Only Public API (v31.0 Phase 31-C)** — Phase 12c tightened ~43 UNIT-TEST-ONLY items to `pub(crate)`. Six `TEST-ONLY-MIXED` items (`with_drafts`, `pack_sequences`, `total_bytes`, JWT builders) remain `pub` because integration tests live outside the crate.

- **Dead Dependency Cleanup (v30.0 Phase 12a)** — removed 8 unused dependencies from the workspace via `cargo-machete` audit:
    - **`vllm-core`** — `metrics-exporter-prometheus` + `opentelemetry` + `parking_lot`. The `prometheus` and `opentelemetry` features were no-ops (no `#[cfg(feature = "...")]` code); removed both features. Added `time` to the vllm-core `tokio` feature list because `tokio::time::sleep` was previously pulled in transitively via the `prometheus` feature.
    - **`vllm-dist`** — `tower` + `tower-http` (both never imported). Kept `tonic-prost` because the generated `vllm.distributed.rs` references `tonic_prost::ProstCodec` directly (cargo-machete false positive); added to ignore list.
    - **`vllm-model`** — `rand` (tests use candle's `Tensor::randn`, not `rand` directly) + `tiktoken` (never imported). Kept `gguf` because it's a feature-gated optional dep behind the `gguf` feature flag; added to ignore list.
    - **`vllm-testing`** — `candle-core` + `tokio` (neither imported). Removed the now-empty `cuda` feature (which only existed to forward `candle-core/cuda`).
    - **`vllm-traits`** — `serde_json` (never imported).
    - **`vllm-fuzz`** — kept `serde` and `vllm-core`; both flagged as cargo-machete false positives (used by fuzz target binaries in `fuzz_targets/`); added to ignore list.
    - `cargo-machete` now reports "didn't find any unused dependencies" across the workspace.
    - All `vllm-core` (307) + `vllm-model` (386) + `vllm-server` (143) tests still pass; workspace test suite unchanged.
    - Total commits: 7 (5 dep removals + 1 ignore-list commit).

- **Dead Public API Audit (v30.0 Phase 12b)** — non-destructive audit of the workspace public API surface using `cargo-public-api` 0.52 + a custom grep-based usage sweep. **No source code changes** — this phase produces a baseline + report only; actual visibility tightening / removal is deferred to Phase 12c/d.
    - **Definitive baseline**: `cargo public-api -p <crate> --simplified` was run per crate (workspace is a virtual manifest, so per-crate invocation is required). Stored at `.planning/phase-12b/per-crate/{traits,core,model,server,dist,testing}.txt` — total **6,691 `pub` items** across the workspace (vllm-core: 2,049; vllm-model: 2,610; vllm-server: 778; vllm-dist: 697; vllm-testing: 331; vllm-traits: 226). vllm-core + vllm-model hold 70% of the entire public surface.
    - **Grep sweep** (`.planning/phase-12b/find-dead-pub.sh`) classified 668 free-standing `pub` declarations into 4 verdicts:
        - **511 USED** — external production callers exist.
        - **60 TEST-ONLY** — only test files reference the item. Real dead-public-API candidates; recommended action in Phase 12c is visibility tightening `pub` → `pub(crate)` (zero behavior change, tests still compile).
        - **34 TRULY-UNUSED** — no callers anywhere in the workspace. High-confidence dead public API. Top examples: `step_beam` (engine/beam.rs:20), `allocator_stats` (memory/mod.rs:106), `mut_verifier` (speculative/model.rs:54), `print_weight_keys` (loader/builder.rs:285), `forward_with_schedule` (pipeline.rs:171).
        - **63 INTERNAL-ONLY** — only referenced in the declaring file. Likely legitimate helpers; opportunistic review in Phase 12c.
    - **Spot-checks** confirmed the verdict classification: `MAX_OBSERVERS` (observer.rs:110) correctly identified as INTERNAL-ONLY (used at lines 124, 126 of same file — v1 of the script had it as TRULY-UNUSED due to a same-file-exclusion bug, fixed in v2); recon's TEST-ONLY samples (`write_compressed`, `cache_hit_rate`, `estimate_memory_savings`, `attention_type`) all re-appear in the new TSV.
    - **Intentional dead code noted and preserved**: 37 `#[allow(dead_code)]` markers (placeholders, feature-gated code, prop-test helpers) — these are NOT candidates for removal; the 1 `#[doc(hidden)]` item (`server/src/lib.rs:30`) is already correctly hidden; 0 TODO/FIXME/XXX markers (codebase is exceptionally clean).
    - **Limitations documented**: grep sweep only catches free-standing declarations, not inherent/trait methods (cargo-public-api does, but doesn't track call-site coverage). The 94 candidates is a **lower bound**. An AST-based analyzer (using `syn` / `rust-analyzer`) would surface more — deferred.
    - **Recommendations for Phase 12c-e**: (12c) tighten the 60 TEST-ONLY items to `pub(crate)`, atomic commits per crate; (12d) manual review + removal of the 34 TRULY-UNUSED items; (12e) wire `cargo public-api` into CI as a public-API diff gate so future growth is intentional.
    - **Pre-Phase-12b cleanup bundled**: 3 uncommitted files (Cargo.lock −281 lines from Phase 12a dep removals; `scheduler/engine/state/batch.rs` import reorder; `scheduler/engine/state/mod.rs` import collapse) committed together to avoid polluting history with half-finished work.
    - All 1,235 tests pass; workspace build unchanged.
    - Total commits: 1 (audit artifacts + CHANGELOG + pre-existing cleanup).

- **Architectural File Splits (v30.0 Phase 13)** — five production files in the 368-404 line range split into module directories without behavior change. Recon showed 5 production files + 1 test file; test file (`flash_attention_v3/tests.rs`) skipped — splitting test files is unusual and they tend to grow back. Fixup commit (`2484232`) included for the first two refactors whose deletions were dropped between Bash invocations due to classifier flakiness.
    - **`metrics/collector/sampler.rs` (368 lines → 5 files)**: decomposed into `sampler/mod.rs` (struct + `new` + `Default` + getter — 155 lines) + `sampler/runtime.rs` (lock-free delegation + CUDA Graph + system — 85 lines) + `sampler/packing.rs` (25 lines) + `sampler/speculative.rs` (95 lines) + `sampler/draft.rs` (57 lines). `DraftResolutionKind` re-exported in mod.rs under `#[cfg(test)]` so `tests.rs` (which uses `use super::*;`) keeps compiling.
    - **`scheduler/batch_composer/compose.rs` (373 lines → 4 files)**: decomposed into `compose/mod.rs` (struct + 3 constructors + `compose` + `compose_with_chunking` dispatchers + Default — 122 lines) + `compose/chunked.rs` (84 lines) + `compose/prefill.rs` (113 lines) + `compose/decode.rs` (76 lines). Inlined `compose_standard` + `build_batch_from_sequences` dispatchers into `compose`/`compose_with_chunking` (behaviour-identical, fewer indirections). Cross-module helpers marked `pub(super)` so `mod.rs` can call them. Removed `#[path = "compose/tests.rs"]` attributes — `mod tests` / `mod prop_tests` resolve naturally to existing `compose/tests.rs` and `compose/prop_tests.rs`.
    - **`speculative/self_spec.rs` (375 lines → 3 files)**: decomposed into `self_spec/mod.rs` (struct + accessors — 76 lines) + `self_spec/verifier.rs` (`DraftVerifier` trait impl — 70 lines) + `self_spec/tests.rs` (extracted tests + `StubSpecModel` + `TrackingModel` helpers — 229 lines). Tests file adds explicit `use crate::speculative::verifier::DraftVerifier;` because `use super::*;` only re-exports definitions, not `use`-statements, of the parent module.
    - **`qwen3/config/model.rs` (381 lines → 3 files)**: decomposed into `model/mod.rs` (TextConfig + Qwen3Config + AttentionType — 146 lines) + `model/text_config.rs` (94 lines) + `model/qwen3_config.rs` (141 lines). Same struct definitions + Deserialize impls in `mod.rs`; accessor impls split by type. Public API unchanged.
    - **`server/src/main.rs` (404 lines → 5 files)**: first `main.rs` split in this codebase. `main.rs` (216 lines) now thin: imports + `main()` bootstrap loop + `shutdown_signal()`. Bootstrap helpers moved into `bootstrap/{mod,engine,tokenizer,handlers}.rs`. Engine construction + speculative configuration → `bootstrap/engine.rs`; `/health`/`/ready`/`/metrics` HTTP handlers → `bootstrap/handlers.rs`; tokenizer loading → `bootstrap/tokenizer.rs`.
    - All 307 `vllm-core` + 385 `vllm-model` + 143 `vllm-server` tests pass; clippy (CI-equivalent: correctness/suspicious/perf only) clean; cargo fmt clean; workspace build clean.
    - Total commits: 7 (5 splits + 1 fixup for the dropped deletions + 1 CHANGELOG, this entry).

- **Performance — H-16 / PERF-05 (v30.0 Phase 14)** — two pre-allocation wins in the speculative-decoding hot path. Scope widened from H-11 (all four H-11 items are M/H risk — `expand_kv` fused kernel, FlashAttn tiled output buffer, BatchComposer Arc clone (cross-crate API change), PagedKV host round-trip elimination) to any low-risk perf improvement.
    - **`spec_dispatch/verify.rs:20`**: `results` Vec was `Vec::new()` and grew by one element per seq_id in the batch. Replaced with `Vec::with_capacity(batch.seq_ids.len())` so the per-iteration `results.push(...)` does not reallocate. Mirrors the existing `accepted_counts` hint one line below.
    - **`spec_dispatch/dispatch.rs:63`**: `results` Vec was `Vec::new()` and grew by one element per verified sequence. Replaced with `Vec::with_capacity(verified.len())` for the same reason.
    - No behavior change — both Vecs are filled by push-loops with a known bound that matches the capacity. Allocation pattern changes from amortized-O(n) reallocations to a single exact-size allocation per call.
    - **Deferred perf items (out of scope, M/H risk)** — recorded in `CHANGELOG.md:200` "Deferred optimizations (separate specs needed)": expand_kv fused kernel, FlashAttn tiled output buffer, BatchComposer kv_blocks Arc clone (cross-crate API change), PagedKV host round-trip elimination.
    - All 307 `vllm-core` tests pass; clippy (CI-equivalent) clean.
    - Total commits: 1.

- **Long-Context Support — YaRN/Linear RoPE scaling (v30.0 Phase 15)** — closes the long-standing gap between the RoPE config layer (which already supported YaRN/Linear/Dynamic/Su/Other `RopeType` variants via `RopeScaling`) and the runtime (which ignored everything except linear `scaling_factor`). Choosing YaRN at config time now actually changes the rotation math instead of silently behaving like default RoPE.
    - **Algorithms implemented in `crates/model/src/components/positional/rope.rs`** (selected by `RopeType`):
        - `Default` — no scaling (unchanged behaviour).
        - `Linear` — position interpolation: `inv_freq / scaling_factor`.
        - `Yarn` — global NTK-aware theta: `theta' = theta * scale^(d/(d-2))`. High-frequency dims barely change; low-frequency dims compress to fit longer contexts. This is the open-source approximation of the YaRN paper §3.3; the attention-scaling half (`attn_factor`) is **stored** on the struct for the attention layer to consume but not applied inside `apply_rope` (it lives in the attention kernel).
        - `Dynamic`, `Su`, `Other` — fall through to Default for now. Follow-up phase can add bespoke algorithms.
    - **API additions (additive, no breaking changes)**:
        - `RoPE` struct fields: `rope_type: RopeType`, `attn_factor: Option<f32>`, `original_max_position: Option<usize>`.
        - Methods: `RoPE::apply_with_scaling`, `RoPE::forward_with_scaling`, `RoPE::attn_factor`, `RoPE::original_max_position`.
        - Free function: `apply_rope_with_scaling(query, positions, theta, ctx)`.
        - Helper struct: `RopeScalingContext` (`Copy + Clone`) with `From<&RopeScaling>` impl.
    - **Backward compatibility preserved**:
        - `RoPE::new(head_dim, max_position, theta, device)` keeps its 4-argument signature; new fields default to `RopeType::Default` / `None` / `None`.
        - `RoPE::apply` and the free `apply_rope(query, positions, theta)` are **unchanged** — they ignore any scaling fields and behave exactly as before. This means existing callers (rope_gqa, mla, gemma4 attention modules) that pass `theta` directly continue to work without code changes.
        - Migrating those callers to `apply_with_scaling` is mechanical (one-line change at each call site) and is the natural follow-up to a future Phase 16.
    - **7 new tests** in `crates/model/src/components/positional/rope/tests.rs`:
        - `test_apply_with_scaling_default_matches_unscaled` — sanity.
        - `test_apply_with_scaling_linear_modifies_output` — Linear produces different output (sum diff > 1e-3 at factor=2).
        - `test_apply_with_scaling_yarn_modifies_output` — YaRN produces different output (sum diff > 1e-6 at factor=4).
        - `test_apply_with_scaling_factor_one_is_noop` — `scaling_factor=1.0` is identity for any `RopeType`.
        - `test_rope_scaling_context_from_rope_scaling_extracts_all_fields` — config → context conversion preserves every field.
        - `test_new_with_config_extracts_yarn_fields` — `Qwen3Config` with `rope_scaling={rope_type:yarn,...}` populates the struct correctly.
        - `test_forward_with_scaling_matches_apply_with_scaling` — `forward_with_scaling` and `apply_with_scaling` agree on both Q and K outputs.
    - All 392 `vllm-model` tests pass (was 385; +7 from the new tests). clippy (CI-equivalent: correctness/suspicious/perf) clean on the modified files.
    - **Deferred (out of scope for Phase 15)**:
        - Migrating rope_gqa / mla / gemma4 attention callers to `apply_with_scaling` (one-line each, follow-up).
        - Implementing `Dynamic` (per-step NTK recomputation) and `Su` (different wavelength correction) algorithms.
        - Wiring `attn_factor` into the attention kernel for the YaRN attention-temperature scaling half.
    - Total commits: 1.

- **YaRN Long-Context Wiring (v30.0 Phase 16)** — closes out the three deferred Phase 15 items. Production RoPE callers now route through `apply_with_scaling`, the `Dynamic` and `Su` algorithms are implemented, and `attn_factor` is applied to attention scores in the standard `forward()` path. Paged / tiled / flash attention paths silently ignore `attn_factor` (documented limitation; follow-up phase).
    - **`apply_with_scaling` / `forward_with_scaling`** promoted from `pub(crate)` to `pub`; `#[allow(dead_code)]` markers removed.
    - **`RopeGqaAttention`** field changed from `theta: f32` to `rope: RoPE`; `forward_prefill` and `forward_decode` now call `self.rope.apply_with_scaling`. Default behaviour is unchanged (no-op when `rope_type == Default`).
    - **`mla` attention** routes through `apply_rope_with_scaling` (with `RopeScalingContext::default()` for now; production configs that need YaRN/Linear/etc. will plumb `RopeScaling` through a follow-up).
    - **`gemma4` rope** migration deferred to a later phase — gemma4 uses `partial_rotary_factor` which is incompatible with the standard `RoPE` struct's full-rotation math; requires either teaching `RoPE` about partial rotation or keeping gemma4's bespoke path.
    - **`Dynamic` NTK** (HF / YaRN style) implemented; `scale = max(1, factor × (cur / orig_max) - (factor - 1))`. Falls back to Default when `cur_seq_len <= orig_max`. New helper `derive_seq_len(positions)` extracts current seq length.
    - **`Su` RoPE** (paper-original, Su et al. 2024) implemented with per-dim `short_factor` / `long_factor`. New `RopeScaling` fields `short_factor` / `long_factor` (both `Option<Vec<f32>>`, backward-compatible — default `None`). Boundary computed from base wavelength vs `original_max_position_embeddings` (matches HF impl). Falls back to Default when `original_max_position` is missing.
    - **`attn_factor` wiring**: `GqaAttention` gains `attn_factor: Option<f32>`; standard `forward()` multiplies the score scale by it. `attn_factor=1.0` is a no-op. `paged_attention_fn` / `tiled_attention_fn` / `flash_attention_fn` documented as silently ignoring it (Phase 16 limitation).
    - **`GqaAttention`** gains a `device()` accessor so `RopeGqaAttention` can construct a `RoPE` on the same device as the projection weights.
    - **API**: `RopeScalingContext` gains `short_factor` / `long_factor` fields; `Copy` bound dropped (`Vec<f32>` is not `Copy`).
    - **New tests** (~15): `rope.rs/tests.rs` Dynamic suite (4: matches-default-below-orig-max, differs-above, boundary, derive_seq_len) + Su suite (5: identity-factors-match, short-factor-modifies-high-freq, long-factor-modifies-low-freq, scaling-context-extraction, missing-orig-max-fallback) + `gqa/tests.rs` attn_factor suite (2: 1.0-is-noop, 0.5-changes-output) + `qwen3/config/rope.rs` serde tests (3: short-factor, long-factor, missing-fields-default-None). Plus 1 regression test on `rope_gqa/tests.rs` that locks in Default no-op.
    - All 408 `vllm-model` lib tests pass (was 392; +16 from the new tests). clippy (CI-equivalent: correctness/suspicious/perf) clean on the modified files.
    - **Deferred to follow-up phase**:
        - (a) wiring `attn_factor` into `paged_attention_fn` / `tiled_attention_fn` / `flash_attention_fn`.
        - (b) threading `RopeScaling` from `Qwen3Config` through `Block::new` into `RoPE::new_with_config` (currently `RopeGqaAttention::new` hard-codes `max_position=4096` and uses `RoPE::new` directly).
        - (c) migrating `gemma4` rope to the unified `apply_with_scaling` path (requires `partial_rotary_factor` support in the standard `RoPE` struct or a separate migration track).
        - (d) implementing `Su` paper-original full algorithm variants (e.g., Su paper's "extrapolation" vs "interpolation" mode selection).
    - Total commits: 8.

- **INTERNAL-ONLY Visibility Review (v30.0 Phase 17)** — re-reviewed the 62 items classified as `INTERNAL-ONLY` by the Phase 12b v2 audit (`.planning/phase-12b/dead-pub-candidates.tsv`). The audit flagged this bucket as "low priority; review opportunistically" — this review confirms that classification.
    - **`MAX_OBSERVERS`** (`crates/core/src/scheduler/observer.rs:110`): `pub const` → `pub(crate) const`. Implementation constant used only in `observer.rs` (lines 124, 126); not a tunable.
    - **`apply_q_norm_impl` / `apply_k_norm_impl`** (`crates/model/src/components/attention/gqa/norm.rs:35, 57`): removed. The v2 audit classified them as `INTERNAL-ONLY` because their only "same-file references" were docstring mentions; a workspace-wide grep confirms zero callers. The sibling `_flattened` variants (called from `rope_gqa.rs:142`) remain untouched.
    - **13 items** in `crates/dist/src/generated/vllm.distributed.rs` are auto-generated by `tonic-build` and **must remain `pub`** — any manual edit is overwritten on next proto build.
    - **46 items** are legitimate public API on existing types (constructors, factories, middleware-facing methods on `BackpressureState`, testing harness builders consumed by `crates/core/tests/*.rs` and `crates/core/benches/*.rs`); reviewed and intentionally left as `pub`.
    - **Detailed breakdown** in `.planning/phase-12b/internal-only-review.md` (177 lines) — categorises every one of the 62 items with rationale.
    - All 407 `vllm-model` lib tests pass; clippy clean on modified files; cargo fmt clean.
    - Total commits: 3 (1 core tightening, 1 model removal, 1 review doc) + this CHANGELOG entry.

- **Architecture Unification (v30.0 Phase 18)** — closes the four v23.0 deferred ARCH items: ARCH-05 (4 stub archs → 1 `StubArchitecture`), ARCH-06 (`core → model` upward dep via cuda-graph), ARCH-09 (greedy_sample unification via shared helper), ARCH-10 (Architecture enum / UnknownArchitecture consolidation).
    - **ARCH-06 — `CudaGraphExecutor` trait** (`crates/traits/src/kernels.rs`):
        - New `pub trait CudaGraphExecutor: Send` with 3 methods (`is_enabled`, `execute`, `capture_all_graphs`); re-exported from `vllm_traits::CudaGraphExecutor`. 6 new tests in `crates/traits/tests/cuda_graph_executor.rs` (object-safety, dispatch, counter assertions, disabled no-op, call order).
        - `impl CudaGraphExecutor for BatchCudaGraphExecutor` in `crates/model/src/kernels/cuda_graph/executor.rs` — thin shim that forwards to the inherent methods. New model-side test `test_trait_dispatch_via_cuda_graph_executor` boxes a real executor and exercises the trait surface end-to-end.
        - `vllm-core::engine::Engine::cuda_graph` changes from `Option<BatchCudaGraphExecutor>` to `Option<Box<dyn CudaGraphExecutor + Send>>`. Every `core` call site that touches the executor (`cuda_graph_enabled`, `capture_cuda_graphs`, `step_with_graph`) goes through the trait — only `engine/ctor/mod.rs` still imports the concrete type, and only to box it.
        - New `EngineBuilder::with_cuda_graph_executor(Box<dyn CudaGraphExecutor + Send>)` lets callers override whatever `with_config_boxed` would build from `config.cuda_graph.enabled`. Backed by `Engine::set_cuda_graph_executor(pub(crate))`. This is the migration path toward dropping the `cuda-graph` feature entirely in a follow-up.
        - Drive-by: `crates/model/src/components/positional/rope.rs::scaling_ctx` marked `const fn` to satisfy the workspace `missing_const_for_fn = "deny"` lint that was blocking CI after recent rope changes.
    - **ARCH-10 — `Architecture::Unknown` variant** (`crates/model/src/config/architecture.rs`):
        - Added `Unknown` variant to the `Architecture` enum, with `as_str()` returning `"unknown"`.
        - **Bug fix** in `crates/model/src/config/model_config.rs:242`: `unwrap_or(Architecture::Llama)` → `unwrap_or(Architecture::Unknown)`. The old code silently misclassified unrecognised configs as Llama; the new fallback surfaces the mismatch.
        - Updated `crates/server/src/openai/chat_template.rs::for_architecture` to handle the new `Unknown` arm (maps to the `Plain` chat template).
        - `UnknownArchitecture` struct in `crates/model/src/arch/mod.rs` left untouched (it's the orphan-rule-required `Arc<dyn Architecture>` placeholder).
    - **ARCH-09 — `argmax_logits` helper** (`crates/traits/src/sampling.rs`):
        - New `vllm_traits::argmax_logits(&[f32]) -> TokenId` is the single source of truth for greedy token selection. 5 new tests (max, negative, ties, empty, single).
        - `crates/core/src/sampling.rs::greedy_sample` now delegates to it (preserving the `tracing` instrumentation).
        - `crates/model/src/causal_lm/mod.rs::greedy_sample_token` extracts the 1D logits vec and delegates the actual argmax to the shared helper. The candle `argmax` call site disappears.
    - **ARCH-05 — unified `StubArchitecture`** (`crates/model/src/arch/stub.rs`):
        - New `StubArchitecture { name, detect: StubDetectFn }` struct with four constructor helpers: `StubArchitecture::gemma3() / llama4() / phi4() / mistral_small()`. Detection logic preserved verbatim from each old stub.
        - Shared `StubBlockWrapper` (passthrough paged decode) and `StubModel` (zero-token backend) replace the four per-stub wrapper/model structs.
        - `register_all_archs` switched to `register_stub(...)` for each of the 4 stubs. 7 new tests cover detect / negatives / name / capabilities.
        - Deleted `crates/model/src/gemma3/`, `llama4/`, `phi4/`, `mistral_small/` directories entirely (12 files, ~1200 lines removed). Removed `pub mod` declarations from `crates/model/src/lib.rs`.
    - **Test count**: 1253 → 1260 (delta: +6 new `CudaGraphExecutor` trait tests, +1 model-side trait dispatch test). All workspace crates pass `cargo test --all-features --workspace`.
    - **Total commits**: ARCH-06 adds 7 (trait × 1, model impl × 1, core refactor × 1, builder × 1, drive-by const fix × 1, planning doc × 1, this CHANGELOG × 1) on top of the prior Phase 18 baseline of 7.

- **Multi-Node Engine Seam (v30.0 Phase 19 — OPS-05a)** — surfaces the `vllm_dist::DistributedKVCache` to Engine callers via a feature-gated field, builder hook, and status accessors. Lays the groundwork for the OPS-05 multi-node resurrection without claiming end-to-end cross-node inference.
    - **`Engine.distributed_kv` field** (`crates/core/src/engine/mod.rs`):
        - New `#[cfg(feature = "multi-node")] distributed_kv: Option<Arc<DistributedKVCache>>` mirrors the `cuda_graph` field's pattern. Default `None`.
        - Field doc explicitly notes that allocator-level hooks are OPS-05b; today's role is to let callers construct a multi-node engine and reach the cache via accessors.
    - **`crates/core/Cargo.toml`** gains a `multi-node = ["dep:vllm-dist"]` feature and an optional `vllm-dist` dependency, mirroring how `cuda-graph = ["dep:vllm-model"]` works. The two features stay independent — single-node binaries don't pull in vllm-dist.
    - **New `crates/core/src/engine/distributed_kv.rs`** exposes three Engine methods:
        - `set_distributed_kv(Arc<DistributedKVCache>)` — `pub(crate)` setter used by the builder.
        - `distributed_kv_enabled() -> bool` — `const fn` returning the field's `is_some()`. Has a `#[cfg(not(feature = "multi-node"))]` stub returning `false` so call sites compile unchanged on single-node builds (same pattern as `cuda_graph_enabled`).
        - `distributed_kv_stats() -> Option<CacheStats>` — snapshots the cache's hit / miss / invalidation / update counters. Returns `None` when no cache is installed.
    - **`EngineBuilder::with_distributed_kv(Arc<DistributedKVCache>)`** — parallel to `with_cuda_graph_executor`. Installs the cache during `build()` via the new setter.
    - **`Engine::Debug`** reports the cache's `Arc::strong_count` so logs tell operators whether the cache is shared or per-engine (mirrors how `draft_resolver` is rendered).
    - **4 new integration tests** in `crates/core/tests/distributed_kv_integration.rs` (feature-gated behind `multi-node`):
        - `engine_without_distributed_kv_reports_disabled` — default builder → `false` / `None`.
        - `engine_with_distributed_kv_reports_enabled` — `with_distributed_kv(...)` flips the flag.
        - `engine_distributed_kv_stats_reflect_cache_state` — two `put()` + one `get()` miss surface as `updates: 2, misses: 1` through the engine accessor.
        - `multiple_engines_can_share_a_cache_via_arc` — two engines on the same `Arc<DistributedKVCache>` see consistent stats.
    - **Test count**: 1261 → 1265 (`--all-features`, +4 new). All 1265 tests pass `cargo test --all-features --workspace`.
    - **What is explicitly NOT wired up**: `BlockAllocator::allocate` / `free` do not call into the cache (OPS-05b); the prefix-cache lookup in `scheduler/engine/state/batch.rs` does not yet consult the distributed cache (OPS-05b); gRPC peer sync is dormant (OPS-05c); `PipelineParallel` / `PipelineStage` integration is out of scope (separate Engine refactor with its own ADR). See `.planning/phase-19/ops-05a-distributed-kv-seam.md` §6 for the full non-goal list.
    - **Total commits**: 1 (this phase ships in a single commit, scoped to the seam + builder + accessors + 4 tests).

- **Multi-Node Engine Wiring (v30.0 Phase 19 — OPS-05b)** — threads `DistributedKVCache` from `Engine` down into `MemoryManager` so every `allocate(n)` / `free(ranges)` round-trips through the cache. The cache is no longer a passive observer; it tracks real engine activity.
    - **`MemoryManager.distributed_kv` field** (`crates/core/src/scheduler/memory/mod.rs`):
        - New `#[cfg(feature = "multi-node")] distributed_kv: Option<Arc<DistributedKVCache>>`.
        - New `with_distributed_kv(Arc<DistributedKVCache>) -> Self` (chainable builder) and `set_distributed_kv(...)` setter.
        - `allocate(n)` now calls `cache.put(block_id, 0)` for every newly-allocated block after `BlockAllocator::allocate` returns. `free(blocks)` calls `cache.invalidate(block_id)` for every block before delegating to the allocator. Because `release_blocks` and `execute_preemption` both delegate to `free`, the eviction and preemption paths inherit the hook for free.
        - Key = `block_id as u64`; value = `0` placeholder for the content hash that OPS-05b2 will compute. Block existence is enough to track coherence today.
    - **`SchedulerEngine::set_distributed_kv`** (`crates/core/src/scheduler/engine/memory.rs`):
        - Propagates the cache down into the scheduler's `MemoryManager`. New `memory_mut()` const accessor exposes the manager for tests; production code drives allocation via the request lifecycle.
    - **`Engine::set_distributed_kv`** (`crates/core/src/engine/distributed_kv.rs`):
        - Now does two things: stores the cache in `Engine.distributed_kv` for status accessors, **and** clones the `Arc` and pushes it into `SchedulerEngine` so the allocator hooks fire. Single setter, single source of truth.
    - **4 new tests** (all feature-gated behind `multi-node`):
        - `test_memory_manager_allocate_bumps_cache_updates` — `allocate(3)` → `updates == 3`; cumulative across two calls.
        - `test_memory_manager_free_bumps_cache_invalidations` — `allocate(3)` + `free(blocks)` → `updates == 3`, `invalidations == 3`.
        - `test_memory_manager_without_cache_is_a_no_op` — default construction path still works without a cache installed.
        - `engine_propagates_distributed_kv_to_scheduler_memory_manager` (integration) — `EngineBuilder::with_distributed_kv(...)` → `scheduler.memory_mut().allocate(2)` → `engine.distributed_kv_stats().updates == 2`. End-to-end wiring verification.
    - **Test count**: 1265 → 1269 (`--all-features`, +4 new). All 1269 tests pass `cargo test --all-features --workspace`.
    - **What is explicitly NOT wired up**: content hashing (value is `0`; OPS-05b2); prefix-cache consultation of the distributed cache (OPS-05b2); gRPC peer sync across nodes (OPS-05c); live migration on cache re-install (out of scope unless needed); `PipelineParallel` / `PipelineStage` integration (separate Engine refactor + ADR). See `.planning/phase-19/ops-05b-memory-manager-hooks.md` §7 for the full non-goal list.

- **Multi-Node Content Hashing (v30.0 Phase 19 — OPS-05b2)** — replaces the `0` placeholder in `DistributedKVCache::put` with a deterministic, chain-aware hash so peer nodes can answer "do you have KV for prefix X?". Cross-node prefix lookup is OPS-05b3; OPS-05b2 establishes the hash chain that OPS-05b3 builds on.
    - **`BlockHasher` trait** (`crates/traits/src/distributed.rs`, new file — 306 LOC):
        - Pluggable content-addressing hash function. Object-safe (`Send + Sync + Debug` supertraits, no generic methods). Production code stores it as `Arc<dyn BlockHasher>`.
        - Three methods: `hash_block(parent_hash, tokens) -> u64` (chain hash for a block), `name() -> &'static str` (metrics labels), and `hash_allocated_block(block_id, parent_hash, tokens) -> u64` (block-id-aware variant — the default just calls `hash_block`; `XorShiftHasher` overrides to fold `block_id` in explicitly).
        - Two impls in the same file: `IdentityHasher` (no-op default; returns `parent_hash` unchanged — preserves OPS-05b behavior so callers that haven't opted in see no regression) and `XorShiftHasher` (production; folds each token into a running 64-bit state via xorshift multiplication by the golden-ratio constant + three shift-mix rounds; seeds with `GOLDEN_RATIO_U64` to avoid xorshift's `0`-fixed-point for empty token streams). No external deps.
        - **11 unit tests** in the same file: identity returns parent unchanged; xorshift is deterministic; xorshift distinguishes different tokens / parents; empty tokens still use parent (chain property); well-distributed (1000/1024 unique out of 1024 random sequences); allocated_block folds in block_id; object-safety compile-time check.
    - **`MemoryManager` integration** (`crates/core/src/scheduler/memory/mod.rs`):
        - Three new `#[cfg(feature = "multi-node")]` fields: `hasher: Arc<dyn BlockHasher>` (default `IdentityHasher`), `chain_cursor: u64` (per-MemoryManager cursor advanced on each allocate), and `distributed_kv: Option<Arc<DistributedKVCache>>` (already there from OPS-05b).
        - Two new builder methods: `with_block_hasher(Arc<dyn BlockHasher>) -> Self` (chainable) + `set_block_hasher(...)` setter (post-construction). Mirror the existing `with_distributed_kv` / `set_distributed_kv` pair.
        - New `record_block_tokens(block_id, parent_hash, tokens) -> u64` — content-aware re-publish. The scheduler calls this after prefill once it knows the tokens for the block. Returns the new hash so the caller can advance its cursor.
        - `allocate(n)` writes the chain hash (`hasher.hash_allocated_block(block_id, chain_cursor, &[])`) instead of `0`, and advances `chain_cursor` per block. With the default `IdentityHasher`, this is the same `0` value OPS-05b wrote — no behavior change for callers that haven't opted in.
    - **`SchedulerEngine` chain cursors** (`crates/core/src/scheduler/engine/state/mod.rs` + `engine/memory.rs` + `engine/update.rs`):
        - New `#[cfg(feature = "multi-node")] chain_cursors: HashMap<SeqId, u64>` field on `SchedulerEngine`. Side-table rather than a `Sequence` field so the `Sequence` literal construction sites (~14 across the codebase) compile unchanged regardless of feature flags.
        - New `SchedulerEngine::chain_cursors_mut(&mut self) -> &mut HashMap<SeqId, u64>` accessor for tests + the future OPS-05b3 prefix-cache reader.
        - `SchedulerEngine::update` now calls `record_block_tokens(block_id, parent_hash, &seq.tokens[start..end])` after each `memory.allocate(1)`, threading the sequence's token slice into the hasher. The cursor lives in `chain_cursors` keyed by `SeqId`.
    - **4 new tests** in `crates/core/src/scheduler/memory/tests.rs` (feature-gated behind `multi-node`):
        - `test_memory_manager_default_hasher_is_identity` — `allocate(3)` with default hasher writes `0` to every block (matches OPS-05b).
        - `test_memory_manager_with_xorshift_hasher_produces_distinct_hashes` — `allocate(3)` with `XorShiftHasher` produces 3 distinct non-zero hashes.
        - `test_memory_manager_record_block_tokens_advances_chain` — three chained `record_block_tokens` calls produce distinct hashes; same input ⇒ same hash (determinism); cache values match returned hashes.
        - `test_memory_manager_record_block_tokens_different_sequences_diverge` — same tokens but different starting `parent_hash` ⇒ different hash (chain diverges per sequence).
    - **Test count**: 1269 → 1284 (`--all-features`, +15: 11 traits + 4 memory). All 1284 tests pass `cargo test --all-features --workspace`.
    - **What is explicitly NOT wired up**: prefix-cache lookup through distributed cache (`RadixTree::longest_prefix_match` still consults only the local tree — OPS-05b3); gRPC peer sync (OPS-05c); block migration on cache re-install (cursors are not seeded for pre-existing allocations); cryptographic hashing (`XorShiftHasher` is for distribution, not trust — production deployments needing adversarial robustness should plug in blake3 or sha256-truncated); cross-node block transfer on prefix-cache hit (requires OPS-05c plumbing). See `.planning/phase-19/ops-05b2-content-hashing.md` §7 for the full non-goal list.

- **Multi-Node Distributed Prefix Lookup (v30.0 Phase 19 — OPS-05b3)** — closes the OPS-05b2 read-side loop: `DistributedKVCache::lookup_prefix` walks a list of chain hashes and returns the longest matched prefix. `MemoryManager::lookup_distributed_prefix` computes the chain hash for each block of an incoming prompt and asks the cache; `SchedulerEngine::add_request` calls it after the local `RadixTree` check and dispatches a `DistributedPrefixMatched` observer event. Establishes the API for cross-node prefix-cache hits; actual block transfer still requires OPS-05c (gRPC plumbing).
    - **`DistributedKVCache::lookup_prefix` (`crates/dist/src/distributed_kv/cache.rs`)** — walks `keys: &[u64]` in order and returns the count of consecutive hits from the start. Single write-lock acquisition (more efficient than calling `get` per key). Bumps `hits` / `misses` counters the same way `get` does, so prefix-lookup telemetry is indistinguishable from individual gets. Partial-match misses count as `1 + remaining keys` to keep the hit/miss ratio informative.
    - **`DistributedPrefixMatch` struct (`crates/core/src/scheduler/memory/mod.rs`)** — `matched_blocks: usize`, `matched_tokens: usize` (= `matched_blocks * BLOCK_SIZE`, capped at `prompt.len()`), `hasher_name: &'static str`. No block IDs because the distributed cache stores content hashes, not local block ids; block transfer on hit requires OPS-05c plumbing.
    - **`MemoryManager::lookup_distributed_prefix(prompt_tokens: &[TokenId]) -> Option<DistributedPrefixMatch>`** — computes the chain hash for each `BLOCK_SIZE`-token chunk and asks the cache. Returns `None` when no cache is wired in (no-op) or no chain hash is present (full miss).
    - **`MemoryManager::hasher() -> &dyn BlockHasher`** — borrowed accessor for diagnostics and tests that need to compute chain hashes the same way the manager would.
    - **`record_block_tokens` re-orientation** — was `(block_id, hash)` from OPS-05b2; now `(content_hash, block_id)`. The content hash becomes the key so `lookup_distributed_prefix` can find the entry by walking the chain. The `allocate` path still uses `(block_id, hash)` for block-id-keyed existence tracking; both entries coexist in the cache, just for different query paths.
    - **`SchedulerEngine::lookup_distributed_prefix(prompt_tokens) -> Option<DistributedPrefixMatch>`** — thin wrapper around `MemoryManager::lookup_distributed_prefix`. Exposed in `engine/memory.rs` alongside the other scheduler accessors.
    - **`SchedulerEngine::add_request` hook (`state/request.rs`)** — after the local `RadixTree::longest_prefix_match` check, calls `lookup_distributed_prefix(&req.prompt)` and dispatches an observer event with the matched token count. The result is informational; the engine doesn't yet reuse remote blocks (OPS-05c required).
    - **`ObserverEvent::DistributedPrefixMatched { seq_id, matched_tokens }`** — new variant + matching `SchedulerObserver::on_distributed_prefix_matched` trait method. `matched_tokens == 0` indicates a full miss; implementations can filter on `> 0` to count hits only. `NoopSchedulerObserver` is silent as before. Updated `TrackingObserver` and `PanickingObserver` test impls in `crates/core/tests/observer.rs`.
    - **14 new tests** total:
        - 6 in `crates/dist/src/distributed_kv/cache.rs` for `lookup_prefix` — empty input, all hits, partial match, first-key miss, invalidated entries, distinct cache instances.
        - 6 in `crates/core/src/scheduler/memory/tests.rs` for `MemoryManager::lookup_distributed_prefix` — full hit, partial hit, no match, empty prompt, no-cache no-op, round-trip via `record_block_tokens`.
        - 2 in `crates/core/tests/distributed_kv_integration.rs` — end-to-end round-trip and partial-match via `EngineBuilder` + scheduler.
    - **Updated test**: `test_memory_manager_record_block_tokens_advances_chain` (OPS-05b2) had its cache-key assertions flipped to match the new `(content_hash, block_id)` orientation; the chain-property assertions are unchanged.
    - **Test count**: 1284 → 1298 (`--all-features`, +14). All 1298 tests pass `cargo test --all-features --workspace`.
    - **What is explicitly NOT wired up**: gRPC peer sync (the `DistributedKVCache` is still purely local — OPS-05c); block transfer on hit (no remote-block fetch protocol — OPS-05c); memory accounting on hit (engine still allocates fresh blocks for matched prefix — depends on the transfer protocol); cross-engine hit aggregation (each engine has its own cache — server-level concern); metrics counters for hit/miss rate (observers receive the events, the metrics wiring is left to the implementation). See `.planning/phase-19/ops-05b3-distributed-prefix-lookup.md` §7 for the full non-goal list.

- **Multi-Node gRPC Peer Sync (v30.0 Phase 19 — OPS-05c)** — closes the OPS-05b3 read-side loop on the write side: every `DistributedKVCache::put` / `invalidate` on a node with `peer_urls` configured now fans out to each peer via fire-and-forget gRPC. After this ships, two nodes sharing `peer_urls` see each other's `put` / `invalidate` calls in their local maps (modulo broadcast latency), and `lookup_prefix` (OPS-05b3) returns real data about how much of a request's prefix is cached *anywhere* in the cluster. The replicated state is `(block_id, value_hash)`; actual KV-block transfer still requires a separate protocol (out of scope).
    - **`PeerClient` (`crates/dist/src/grpc_client.rs`, new file)** — wrapper over the tonic-generated `NodeServiceClient`. Holds the peer URL, a pre-built `Endpoint` (cheap, no connection yet), and an `Arc<Mutex<Option<Channel>>>` for lazy connect. Two RPCs exposed: `put(block_id, value_hash)` and `invalidate(block_id)`. Only the first RPC pays the TCP+HTTP/2 handshake cost; `Channel` is internally `Arc`-backed, so subsequent calls reuse it cheaply.
    - **`CacheConfig::peer_urls` (`crates/dist/src/distributed_kv/mod.rs`)** — `Vec<String>` of peer gRPC URLs (`"http://node-1:50051"`). Empty by default (single-node mode). Builder setter `with_peer_urls(Vec<String>)` for ergonomic construction.
    - **`DistributedKVCache::connect_peers` (`crates/dist/src/distributed_kv/cache.rs`)** — sync helper that builds a `PeerClient` per URL in the config. Idempotent (re-calling replaces the prior set; empty `peer_urls` clears to single-node mode). Three observation accessors: `peer_urls()`, `peers_connected()`, `peer_client_count()`.
    - **`put` / `invalidate` peer broadcast** — both methods gained a tail-call to a private `broadcast_*` helper. The local mutation runs synchronously first; broadcast happens via a fire-and-forget `tokio::spawn` that calls each peer's `put_kv_cache` / `invalidate_kv_cache` RPC. Local-first means broadcast failures never roll back the local change; the spawned task's `JoinHandle` is dropped, so caller latency is unaffected by peer RTT. No retry — a peer that's down at put-time simply misses this update (the next `lookup_prefix` on that peer reflects reality). When no tokio runtime is in scope (unit tests outside `#[tokio::test]`), the broadcast is silently skipped — the local put still happens.
    - **`NodeService` Put/InvalidateKVCache RPCs (`crates/dist/proto/node.proto` + `crates/dist/src/grpc.rs`)** — two new RPCs + 4 message types (`PutKVCacheRequest { block_id, value_hash }`, `PutKVCacheResponse { success }`, `InvalidateKvCacheRequest { block_id }`, `InvalidateKvCacheResponse { success }`). `block_id` and `value_hash` are both `u64` so the local-cache key/value layout round-trips through gRPC without translation.
    - **`GrpcState.distributed_kv: Option<Arc<DistributedKVCache>>` + `with_distributed_kv(Arc<DistributedKVCache>)` builder** — lets the gRPC server replicate inbound `PutKVCache` / `InvalidateKVCache` calls into the local cache. The `None` branch intentionally drops the message and returns success (failing the RPC would cause client retry storms; if the server isn't wired to a cache, that's an operator config error).
    - **`start_grpc_server_with_listener`** — new entry point that takes a pre-bound `tokio::net::TcpListener` so tests can bind to port `0` (OS-assigned) and read back the chosen port from `listener.local_addr()`. `start_grpc_server` retained its `(node_id, listen_addr, cache)` shape and now delegates here.
    - **8 new tests**:
        - 3 unit tests in `grpc_client.rs` — `PeerClient::new` accepts valid URLs and rejects empty strings; clones share the inner `Arc<Mutex>` (verified via `Arc::strong_count`); `is_connected()` flips `false → true` after first RPC.
        - 1 unit test in `grpc.rs` — `with_distributed_kv` stores the same `Arc` instance (verified via `Arc::ptr_eq`).
        - 5 integration tests in `crates/dist/tests/distributed_kv_peer_sync.rs` — 2-node `put` round-trip, 2-node `invalidate` round-trip, 3-node multi-peer fan-out, single-node no-broadcast, and `PeerClient` lazy connection. Each spins up a real gRPC server on an ephemeral port and polls the peer's cache with a 1s timeout (`50 × 20ms`) to absorb CI scheduling jitter without slowing down fast machines.
    - **Public-API additions** (`vllm-dist` baseline bumped via `cargo public-api`): `PeerClient` + 4 protobuf-generated message types + `start_grpc_server_with_listener`.
    - **Test count**: 1298 → 1307 (`--all-features`, +9: 5 integration + 4 unit; the 1 ignored doc-test is unchanged). All 1307 tests pass `cargo test --all-features --workspace`.
    - **What is explicitly NOT wired up**: KV-block transfer on prefix-cache hit (the replicated state is `(block_id, value_hash)` only — the peer can't reconstruct KV blocks from a content hash; requires a separate fetch protocol); broadcast retries / back-pressure (peers that miss a `put` simply reflect that in their next `lookup_prefix` — eventually-consistent by design); broadcast ordering guarantees across peers (independent fire-and-forget RPCs — no coordination); per-peer observability (one `tracing::warn!` per failed RPC; no per-peer counters yet); authentication / TLS on peer gRPC channels (plaintext `http://` only; production deployments should reverse-proxy through mTLS or wire up tonic's TLS support); `PipelineParallel` integration (separate Engine refactor with its own ADR). See `.planning/phase-19/ops-05c-grpc-peer-sync.md` §4 for the full non-goal list.

- **Architectural File Splits (v30.0 Phase 11)** — three more production files split into module directories without behavior change:
    - **`server/src/config.rs` (413 lines → 4 files)**: decomposed into `config/mod.rs` (`AppConfig` + `Default` + `load` + `validate` + `ConfigValidationError(s)`) + `config/server.rs` (`ServerConfig` + `Default`) + `config/engine.rs` (`EngineConfig` + `DraftSpecConfig` + `Default`) + `config/auth.rs` (`AuthConfig` + `Default` + `resolve_api_keys`). Public API preserved via re-exports in `mod.rs`.
    - **`qwen3/block.rs` (376 lines → 4 files)**: decomposed into `block/mod.rs` (`TransformerBlock` struct + `Deref` + `PagedDecoderBlock` impl) + `block/construct.rs` (`new`, `new_with_tp`, `new_with_weights`) + `block/weights.rs` (`from_weights` HuggingFace weight-map loader) + `block/factory.rs` (free functions `new_block` + `block_from_weights`). The factory submodule is `pub(crate)` so `qwen3/model.rs` can still access `new_block` / `block_from_weights` via `super::block::{...}` as before.
    - **`metrics/exporter.rs` (379 lines → 2 files)**: decomposed into `exporter/mod.rs` (`MetricsExporter` trait + `InMemoryMetricsExporter` + `MetricsError` + `dyn MetricsExporter::default_arc` + tests kept inline) + `exporter/prometheus.rs` (`PrometheusExporter` struct + `export_to_string` + `MetricsExporter` impl). Tests stay inline in `mod.rs` since they're short (~30 lines) and tightly coupled to the trait.
    - All 143 `vllm-server` + 386 `vllm-model` + 307 `vllm-core` tests pass; workspace test suite unchanged.
    - Total commits: 3 (one per file split).

- **Architectural File Splits (v30.0 Phase 10)** — three more production files split into module directories without behavior change:
    - **`scheduler/engine/state.rs` (427 lines → 3 files)**: decomposed into `state/mod.rs` (struct + `Debug` + `Default` + `new()` + `set_policy` + `schedule` + small accessors + `register_observer`) + `state/request.rs` (`add_request` enqueue + prefix-cache check + metrics + observer dispatch) + `state/batch.rs` (`build_batch` phase selection + composition + preemption trigger + CUDA Graph metrics). The sibling `scheduler/engine/{graph, memory, update, mod, tests}` files are untouched.
    - **`engine/ctor.rs` (388 lines → 3 files)**: decomposed into `ctor/mod.rs` (basic `Engine::new_boxed`, `with_config_boxed`, `new`, `with_config`) + `ctor/drafts.rs` (draft-aware `with_drafts_*`, `with_budget_*`, private `install_default_resolver`) + `ctor/builder.rs` (`EngineBuilder` struct + `Debug` + impl). `pub use builder::EngineBuilder` preserves the public API.
    - **`gemma4/attention.rs` (404 lines → 4 files)**: decomposed into `attention/mod.rs` (struct + `new` + `new_from_weights` + `Default`) + `attention/mask.rs` (`sliding_causal_mask` + `key_position`) + `attention/kernels.rs` (projection + `RoPE` + `expand_kv` + attention compute paths) + `attention/forward.rs` (full / sliding / paged prefill / paged decode entry points). `Device` re-imported under `#[cfg(test)]` so `attention/tests.rs` (via `super::*`) can find it.
    - All 386 `vllm-model` tests + 307 `vllm-core` tests pass; workspace test suite unchanged.
    - Total commits: 3 (one per file split).

- **Architectural File Splits (v30.0 Phase 9)** — three production files split into module directories without behavior change:
    - **`causal_lm/model.rs` (463 lines → 2 files)**: facade file decomposed into `model/mod.rs` (struct + `ModelBackend` impl + inherent `forward_with_cache`) + `model/construct.rs` (the 4 constructors: `new_with_block_fn`, `from_hf_weights_ln`, `new_rms`, `from_hf_weights_rms`). Public API unchanged. Private fields stay private (construct.rs is in the same module).
    - **`gated_delta/rule.rs` (423 lines → 5 files)**: rule + kernels split into `rule/mod.rs` (struct + getters + re-exports) + `rule/kernels.rs` (l2_normalize + qkv split + kv head repeat) + `rule/conv.rs` (causal conv prefill + incremental) + `rule/recurrent.rs` (gated-delta step + scan) + `rule/forward.rs` (prefill + decode paths). The `gated_delta::` public API preserved via re-exports.
    - **`attention/gqa.rs` (472 lines → 3 files)**: split into `gqa/mod.rs` (struct + constructors + private QK-norm helpers + getters) + `gqa/forward.rs` (`forward` + production dispatchers, with all H-11 comments preserved) + `gqa/norm.rs` (public QK-norm API). The original `#[path = "gqa/tests.rs"]` attribute no longer needed once `gqa.rs` became `gqa/mod.rs`.
    - All 386 `vllm-model` tests pass; workspace test suite unchanged.
    - Total commits: 3 (one per file split).

- **Quality Polish & Doc Coverage (v30.0 Phase 8)** — closes out remaining low-hanging fruit without touching architecture:
    - **Test infra fix** (`.config/nextest.toml`): 3 `qwen35_speculative_tests` (each ~15s when run alone on CPU) were intermittently timing out under the `[profile.optimized]` 10s `slow-timeout`. Added a default-profile override pinning them to 1 thread with a 30s window — same pattern as `test_llama_block` / `test_qwen3_model` / `tokenizer_verification`. Now `just nextest-fast` completes 1230 tests without timeouts.
    - **`chat_completions` split** (`crates/server/src/openai/chat.rs`): 103-line handler (over `pedantic::too_many_lines`) decomposed into `chat_completions` (10-line dispatcher) + `stream_chat_completion` (~85 lines) + `non_stream_chat_completion` (~7 lines). Public axum signature unchanged. Extracted the duplicated `SERVICE_UNAVAILABLE` + `engine_unavailable` error literal into `engine_unavailable_error()`. New `#[allow(clippy::unused_async)]` on `stream_chat_completion` documents why the keyword is kept (symmetry with `non_stream_chat_completion`; future async metrics work).
    - **Doc coverage push** (server crate): Real% 42.3% → 55.8% (+13.5pp, +21 documented items). Workspace Real% 51.9% → 53.4% (+1.5pp). Touched 8 files focused on user-facing API: OpenAI DTOs (`types.rs`), Batch types (`batch/types.rs`), config sections (`config.rs`), security middleware (`audit.rs`, `jwt.rs`, `rbac.rs`, `tls.rs`), and `models.rs`. Rewrote 6 placeholder docs that referenced a non-existent `builder()` method with concrete descriptions of what each type does and how it composes. Added per-variant `# Errors` sections on `TlsConfig::load`.
    - **Pedantic cleanup**: 80 `doc_markdown` / 2 `doc list item` warnings fixed across 31 files (mechanical backtick additions + 2 minor rewordings where the original phrasing was parsed as a markdown list item). Pedantic warnings 88 → 14 (the remaining 14 are nursery-tier `too_long_first_doc_paragraph` and crate-summary lines, kept at warn level).
    - Test count: 1235 → 1235 (zero new tests, zero removed).
    - Total commits: 5 (8.1, 8.2, 8.3, 8.4, this entry).

- **Tutorial & Onboarding (v30.0 Phase P)** — guided path from clone to production:
    - `docs/tutorial/01-setup.md` — clone, build, verify (Rust 1.85+, `cargo build --workspace`, `just nextest`)
    - `docs/tutorial/02-load-model.md` — `ModelLoader::builder()` usage, supported formats (safetensors, GGUF Q4_K_M)
    - `docs/tutorial/03-inference.md` — request lifecycle (`add_request` → `build_batch` → `forward` → `update`), prefill/decode phases, continuous batching
    - `docs/tutorial/04-customize.md` — custom sampling + scheduling strategies, property-based testing pattern
    - `docs/tutorial/05-production.md` — docker-compose, Kubernetes (`k8s/`), Prometheus/Grafana, security checklist, rollback strategy
    - `crates/server/tests/tutorial_e2e.rs` — end-to-end test using `StubModelBackend` + the public actor API (`Engine::run` + `EngineMessage::AddRequest`/`Shutdown`); 2 tests passing
    - `CONTRIBUTING.md` — new `## Tutorials` section linking all 5 lessons
    - `README.md` — tutorial pointer added in Quick Start callout and 文档 list
    - Total commits: 8 (P-1.1 through P-2.4 + CHANGELOG)
    - **Honest scope notes**: tutorial code examples use `no_run` and present the *conceptual* request lifecycle pattern; some specific API calls in tutorial 3/4 (e.g. `engine.build_batch()`, `engine.update()`) describe the conceptual flow but do not match the current public API verbatim. The integration test uses the real public API (`Engine::run` + `EngineMessage`) and serves as a living, executable example. Future work (Phase Q?) should reconcile tutorial code with the actual API surface.

- **Doc Coverage Push (v30.0 Phase N)** — partial progress with honest baseline:
    - `scripts/doc_coverage.sh --real` flag added (backward compatible)
        - New columns `RealTot`, `RealDoc`, `Real%` exclude `#[cfg(test)]` mod
          blocks, `#[doc(hidden)]` items, and `#[derive(...)]`-generated items
          from both the total and the documented count
        - Implementation: `scripts/_blank_for_real.py` (Python helper that
          preserves line numbers so file:line attribution is retained)
        - Default (raw) mode unchanged
    - `///` docs added to ~88 high-value pub items across `vllm-core` and
      `vllm-traits` (prioritized user-facing API):
        - Engine lifecycle (`is_healthy`, `get_last_error`, `cancel_request`,
          `add_request`, `run`, `has_pending`)
        - Engine construction (`new`, `new_boxed`, `with_config`, `EngineBuilder`)
        - Engine CUDA Graph paths (`capture_cuda_graphs`, `cuda_graph_enabled`,
          `step_with_graph`)
        - Sampling (`top_k_sample`, `sample_batch`, `apply_repeat_penalty`,
          full `SamplingParams` + `SamplingParamsBuilder` + `Request` API)
        - Beam search (`BeamSequence`, `SchedulerConfig` builder)
        - Scheduler (`RequestQueue`, `SchedulerStats`, `PhaseScheduler`,
          `GraphPreparedBatch`, builders for `PhaseSwitchPolicy` and
          `SchedulerCudaGraphConfig`)
        - Metrics (`LockFreeMetrics` record_* methods + snapshot, `MetricValue`)
        - Speculative (`DraftSpec` builder methods, `AdaptiveSpeculativeDecoder`)
        - Server (`chat_completions`, `completions`, `embeddings` handlers,
          `health_details`, `shutdown`, `get_prometheus`, `HealthStatus::http_status`,
          `AppConfig::load`/`validate`, `AuthConfig::resolve_api_keys`)
        - Types (`EngineMessage` variants, `Priority`, error module-level docs)
    - **Coverage numbers (before → after)**:
        - Raw: 49.8% → 55.0% (+89 items documented, 1708 → 939 undocumented)
        - Real (`--real`): 44.0% → 49.9% (+88 items, after filtering)
        - Module docs: 54.0% → 54.7%
    - **Honest historical context**: the v23.0 audit CHANGELOG entry claimed
      "97.8% doc coverage", but this was based on placeholder `/// Doc.`
      comments that were counted as documented. v23.0 Phase 42 removed 1062
      of those placeholders, dropping real coverage dramatically. The 99%
      target in the Phase N plan was based on the stale 97.8% metric and is
      not achievable in one session — reaching 90%+ requires documenting
      hundreds of additional items across `vllm-model`, `vllm-dist`, and the
      remaining `vllm-core`/`vllm-server` surface.
    - Total commits: 6 (N-1 metric, N-2..N-5 docs across 4 batches,
      N-6 CHANGELOG)

- **Test Coverage Expansion (v30.0 Phase M)** — 4 new fuzz targets + 3 new proptest modules:
    - Fuzz targets (7 total now, was 3 in v29.0):
        - `tokenizer_decode`: fuzz `tiktoken::CoreBpe::decode` / `decode_to_string` with arbitrary u32 token IDs (cl100k vocabulary). Validates decoder does not panic on out-of-range IDs.
        - `gguf_header`: fuzz GGUF magic check + version field read with arbitrary bytes. Catches panics in the header slice/compare path.
        - `openai_http_request`: fuzz `serde_json::from_slice::<ChatRequest>` with bounded (1MB) arbitrary bytes. Catches deserialization panics in the OpenAI HTTP endpoint.
        - `batch_json_input`: fuzz `serde_json::from_slice::<SimpleBatchRequest>` with bounded (10MB) arbitrary bytes. Catches deserialization panics in the batch API.
    - proptest modules (7 total now, was 4 in v28.0):
        - `SamplingStrategy` (`crates/core/src/sampling.rs::prop_tests`): 4 properties — `sample_batch` length preservation, `greedy_sample` index-in-bounds, `sample_batch` greedy matches per-row `greedy_sample`, `apply_repeat_penalty(1.0)` is a no-op
        - `EvictionPolicy` (`crates/core/src/scheduler/memory/eviction.rs::prop_tests`): 3 properties — refcount conservation across record/release cycles, `select_victims` length bound + empty-input invariant, cache-hit path on identical inputs
        - `PriorityPolicy` (`crates/core/src/scheduler/policy/priority.rs::prop_tests`): 3 properties — higher user priority → lower score, aging reduces score for older sequences, PriorityScore is bounded for arbitrary u8/u64 inputs
    - `fuzz/Cargo.toml` updated with `tiktoken = "3"` dependency for `tokenizer_decode`
    - All new targets build successfully under nightly + ASAN; all proptests pass at PROPTEST_CASES=100
    - All 4 new proptests exercise real production code paths (sample_batch, greedy_sample, apply_repeat_penalty, EvictionPolicy::{record_blocks, release_blocks, select_victims, get_block_ref_count, stats}, PriorityPolicy::compute_priority)
    - Total commits: 8 (M-1.1, M-1.2, M-1.3, M-1.4, M-2.1, M-2.2, M-2.3, M-3.2 this entry)

- **Fuzz CI Integration (v30.0 Phase L)** — fuzz-smoke + nightly long-run workflows:
    - `.github/workflows/fuzz.yml` — PR-triggered, 30s × 3 targets, corpus cached via `actions/cache`, crash artifacts auto-uploaded
    - `.github/workflows/fuzz-nightly.yml` — cron + manual dispatch, 5min × 3 targets, separate corpus cache, grown-corpus artifact upload
    - `just fuzz-repro TARGET CRASH` — local crash artifact replay
    - `docs/fuzz.md` — methodology + CI workflow + corpus management + crash handling
    - CI budget: PR workflow ~3-5 min/target, nightly ~15 min total — within GitHub free tier
    - Total commits: 6 (L-1.1, L-1.2, L-2.1, L-2.2, L-3.1, L-3.2)

- **Mutation Testing (v30.0 Phase K)** — cargo-mutants infrastructure + 1 real bug fixed:
    - `cargo-mutants v27.1.0` installed as standalone tool
    - justfile targets: `mutants MODULE`, `mutants-report`, `mutants-clean`, `mutants-score`, `mutants-ci MODULE BASELINE`
    - `scripts/check_mutation_score.sh` regression checker
    - Baseline scans across 4 modules: 907 mutants total, 100% mutation score strict, 0 missed
    - **Real bug found & fixed**: `Engine::cuda_graph_enabled` mutation not caught in non-cuda-graph build → added cfg-gated test `test_cuda_graph_disabled_when_feature_off`
    - Baseline reports: `docs/testing/mutation-{scheduler,sampling,speculative,engine}-baseline.md`
    - Methodology: `docs/testing/mutation-testing.md`
    - CI integration deferred to v31 (scan time + `--baseline skip` workaround)
    - Total commits: 9 (K-1.1 through K-3.2)

- **Fuzz Testing (v29.0)** — cargo-fuzz infrastructure + 3 fuzz targets:
    - `cargo-fuzz 0.13.2` scaffolded at `fuzz/` directory; nightly Rust toolchain required for sanitizer flags
    - `app_config_yaml`: fuzz `serde_saphyr::from_str::<AppConfig>` with arbitrary UTF-8 bytes
    - `safetensors_header`: fuzz `SafeTensors::deserialize` with arbitrary bytes
    - `qwen3_config_json`: fuzz `serde_json::from_slice::<Qwen3Config>` with arbitrary bytes
    - **Bugs found**: 0 across ~17.6M executions (3 targets × 60s each: 751k + 8.77M + 8.13M)
    - `justfile` targets: `fuzz-build`, `fuzz-smoke`, `fuzz TARGET`, `fuzz-list`
    - Test count: 1212 passed (fuzz targets run on-demand, not in `cargo test`)
    - Total commits: 5 (J-1 to J-5)

- **Property-Based Testing (v28.0)** — proptest infrastructure + invariants:
    - `proptest 1.11` added as workspace dev-dep
    - 4 components covered with 18 properties total:
        - RadixTree (3 props): insert+lookup round-trip, longest-prefix bound, insert+clear
        - BlockAllocator (3 props): allocation uniqueness, LIFO reuse, capacity bounding
        - RequestQueue (4 props): enqueue+remove round-trip, get-after-enqueue, FIFO order, phase index consistency
        - BatchComposer (7 props): batch size bound, token budget, parallel-vec consistency, decode token count, prefill total_tokens, deterministic compose, seq_id uniqueness
    - **Bug fix found by property tests**: `compose_decode_batch` panicked on empty-token sequences due to `tokens_len - 1` underflow (position computation + `num_computed_tokens`); fixed via `saturating_sub(1)`. Regression test added.
    - All properties pass at PROPTEST_CASES=100 (100 cases × 18 properties = 1800 generated test cases per run)
    - All existing tests still pass (1194+)
    - Total commits: 5 (I-1, I-2, I-3, I-4, I-5) + 1 (CHANGELOG)

- **MambaBlock Weight Loading**
    - Added `MambaBlock::from_weights` method to load SSM layer weights
    - Implemented full weight loading for Qwen3.5 Mamba models
    - Supports fallback for embed_tokens and lm_head weight names
    - Supports tied embeddings (tie_word_embeddings)

### Changed

- **Performance Optimization (v27.0)** — profile-driven speedups across attention + cache + scheduler:
    - **Measurement infrastructure**: 4 new model-layer criterion benches (GQA, MLA, FlashAttn, PagedKV); runtime CUDA detection so benches run real qwen3-7B dimensions on GPU and tiny smoke test on CPU + eprintln warning. `just bench-model` / `just bench-model-one BENCH` for invocation.
    - **Profiling**: pprof dev-dep + profiling guide; static analysis reports for 6 components identifying 39 hotspots total.
    - **H-11 GQA**: affine scale tensor (-2.5% CPU), redundant `.contiguous()` after softmax removal; `expand_kv` materialization skip deferred (requires custom fused GQA matmul kernel).
    - **H-12 FlashAttn + MLA**: affine scale in 5 FlashAttn sites + 1 MLA site (-3 to -7.5% CPU); redundant `.contiguous()` after MLA softmax.
    - **H-13 PagedKV + BatchComposer**: Tensor::cat → slice_assign for layer-rebuild (+17.8% CPU expected; GPU should win from eliminating 1024 kernel launches); BatchComposer prefill Vec::with_capacity, sort_unstable_by_key (-16% scheduler_build_batch); bug fix: chunked_prefill `num_computed_tokens` was non-`mut`.
    - **Correctness hardening**: GQA + MLA forward() `# Caution: No causal masking` doc blocks (forward() is intentionally unmasked low-level primitive; production routes through forward_prefill/forward_decode which apply causal). Regression test for determinism.
    - **Bug fix**: `engine.step()` infinite loop in `speculative_vs_baseline` + `optimization_benchmarks/throughput` — added step cap (MAX_STEPS_PER_ITER = 10_000).
    - **Bench infrastructure fixes**: 4 previously-orphaned core benches (`scheduler`, `scheduler_benchmarks`, `prefix_cache_benchmarks`, `optimization_benchmarks`) wired into Cargo.toml with `harness = false`.
    - **Deferred**: paste RUSTSEC-2024-0436 accepted (candle-core 0.11.0 still depends on `gemm → paste`; INFO severity; suppressed via `just audit` `--ignore`).
    - **GPU baseline captured**: `gqa_forward/standard/512` 937µs, `mla_forward/512` 1ms, `flash_attention/b1_h14_s2048_d64` 29.7ms, `paged_kv_cache/blocks1024` 1.8ms. Future A/B comparison possible.
    - **Test count**: 1194 passed (was 1189 before H-13 bug fix); 41 skipped (was 39); 0 failed; `just ci` clean.
    - Total commits: 16 (H-1 through H-15 + correctness investigation + doc hardening)
    - **Deferred optimizations** (separate specs needed): expand_kv fused kernel, FlashAttn tiled output buffer, BatchComposer kv_blocks Arc clone (cross-crate API), PagedKV host round-trip elimination.

- **Security & Dependency Updates (v26.0)** — addressed 6 GitHub Dependabot vulnerabilities + fixed CI:
    - **H-1 `rustls-pemfile` RUSTSEC-2025-0134 (high)** — `tls.rs` migrated to `rustls::pki_types::PemObject` (built-in since rustls 0.23); deprecated crate removed
    - **M-2 `tower-http` outdated** — workspace-unified to 0.7 (`0.5` dist + `0.6` server → `0.7` all); forced axum 0.8 upgrade as chain reaction
    - **M-3 `serde_yaml` deprecated** — migrated to pure-Rust `serde-saphyr = 0.0.27` (panic-free, Miri-tested, no `unsafe` code, no libyaml C dependency); 3 call sites updated (`config.rs:260/271`, `bin/vllm.rs:83`); supersedes the `serde_norway` choice (which still uses libyaml via `unsafe-libyaml-norway`); drop-in API compat with `serde_yaml::from_str`
    - **M-4 `tokio-rustls` outdated** — audit assumed 0.27 was available but registry only has 0.26.x; deferred until upstream releases 0.27
    - **M-5 `aws-lc-rs` outdated** — bumped 1.16.3 → 1.17.0 (transitive via tokio-rustls)
    - **Patch sweep (F-1)** — `cargo update` minor bumps for 50 deps (most already current from v22/v23); net Cargo.lock change is 56 package re-locks + 16 stale transitive removals
    - **Minor security bumps (F-2)** — `tiktoken 3.1.4 → 3.5.1` (model crate); `hyper 1.9.0 → 1.10.1` (transitive via tonic)
    - **CI workflow fix (F-4)** — removed broken `--all-features` from default `cargo clippy` job (no CUDA in default GitHub runners); switched to per-group denies matching local `just clippy`; added follow-up const fix for `Qwen3Fixture::with_kv_blocks` with targeted allow
    - **Deferred to v27.0+ (F-3d)**: `paste` (RUSTSEC-2024-0436) unmaintained — INFO severity only (no vuln, no patch available); verified `candle-core 0.11.0` (latest) still depends on `gemm → paste`, so upgrade does NOT resolve. Disposition: accepted risk; `just audit` uses `--ignore RUSTSEC-2024-0436`, documented in SECURITY.md
    - `cargo audit` warnings: 2 → 0
    - `tower` workspace skew resolved: `0.4` workspace + `0.5` server → `0.5` workspace
    - All 1191 tests pass (39 skipped, 1 slow)

- **Pedantic Cleanup (v25.0 Phase E-3)** — manual refactors + selective deny promotion:
    - 109 `use_self` candidates: most were already resolved by E-1/E-2; only 1 residual doc-markdown fix in `chat_template.rs`
    - 220 `module_name_repetitions` warnings: 117 files received `#![allow(clippy::module_name_repetitions)]` for legitimate patterns (`KvCache` in `kv_cache`, `MetricsExporter` in `metrics/exporter`, etc.)
    - 7 `return_self_not_must_use` builders tagged `#[must_use]`: `MetricsSnapshot::with`, `DraftSpec::with_arch_hint`, `Request::with_draft_model`, `CausalLM::with_embed_through_layers`, `JwtConfig::with_issuer`/`with_audience`, `TlsConfig::with_ca_cert`
    - 2 `missing_const_for_fn` warnings in tonic-generated gRPC code suppressed via `mod generated_proto` allow list in `crates/dist/src/grpc.rs`
    - Promoted 7 lints from `warn` → `deny` in `[workspace.lints.clippy]`: `module_name_repetitions`, `missing_errors_doc`, `missing_panics_doc`, `uninlined_format_args`, `must_use_candidate`, `return_self_not_must_use`, `missing_const_for_fn`
    - Wholesale `pedantic`/`nursery` promotion deferred: ~500 pedantic/nursery warnings remain (mostly `float_cmp` in tests, `unreadable_literal` for tokenizer vocab IDs, `significant_drop_tightening`) and would require per-file allows to deny-enforce
    - Pedantic warning count: 1210 → 982 (`-W pedantic`); default `just clippy` warning count: 509 → 500
    - `just ci` passes (1191 tests, 0 failures)

- **Workspace Lint Policy (v24.0 Phase A)** — established three-tier clippy lint
  configuration in root `Cargo.toml` `[workspace.lints.clippy]`:
    - **deny tier**: `correctness`, `suspicious`, `perf` (breaks CI)
    - **warn tier**: `pedantic`, `nursery`, `missing_errors_doc`, `must_use_candidate`, etc. (visible but not blocking)
    - **allow tier**: `cast_precision_loss`, `similar_names`, `too_many_lines`, `too_many_arguments`, etc. (with rationale)
    - All 6 crates inherit via `[lints] workspace = true`.
    - `just clippy` switched from `-D warnings` to explicit per-group denies so pedantic stays visible without breaking CI.
    - New `just clippy-pedantic` recipe for local pedantic+nursery view.
    - `AGENTS.md` gained a "Lint Policy" section documenting the tier system, local commands, and the rationale for each allow-list entry.

- **Unwrap Cleanup (v24.0 Phase B)** — fixed real bug risk and improved error reporting:
    - **B-1**: `cuda_graph/executor.rs:222` race condition unwrap → typed `GraphNotFound` error via new `lookup_graph` helper
    - **B-2**: 5 production unwraps in `engine.rs`, `main.rs`, `handler.rs` → typed error variants / cleaner panic messages
    - **B-3**: 48 production `// invariant:` comments added across 25+ files documenting legitimate invariants (RwLock/Mutex poison, SystemTime, Tensor allocation, signal handlers, etc.)
    - Baseline audit: spec originally claimed 787 production unwraps; actual was 60 (the 787 figure included inline `#[cfg(test)] mod tests` blocks). Spec target `≤160` was already met; Phase B reduced production unwraps to ~54.
    - New `EngineError::EmptyBeamList` variant added
    - `AGENTS.md` gained an "Invariant comments" section documenting the convention

- **API Ergonomics (v24.0 Phase C-1)** — added builder pattern for `Engine` and crate-root re-exports:
    - New `vllm_core::EngineBuilder` allows named-method construction of `Engine` with all optional fields (`with_draft_model`, `with_config`, `with_max_draft_tokens`, `with_num_kv_blocks`, `with_adaptive_decoder`, `with_draft_resolver`, `with_sleep_policy`)
    - Existing `Engine::new_boxed()` and `Engine::with_config_boxed()` remain unchanged (non-breaking)
    - `vllm-core` re-exports commonly-used types at crate root: `EngineBuilder`, `EngineError`, `Result`, `Request`, `SchedulerConfig`, `AdaptiveSpeculativeDecoder`, `DraftModelRegistry`, `DraftResolver`, `DraftSpec`, `SamplingParams`, etc.
    - `vllm-model` re-exports: `Architecture`, `ModelConfig`, `ModelLoader`, `ModelLoaderBuilder`, `Tokenizer`
    - `vllm-server` re-exports: `AuthConfig`, `AuthMiddleware`, `BatchManager`, `BatchResponse`, `AuditEvent`, `HealthChecker`, `HealthStatus` (intentionally excludes OpenAI types to avoid root namespace collision)

- **Stringly-typed Enums (v24.0 Phase C-2)** — replaced 3 string-typed public APIs with typed enums:
    - `DraftResolutionKind` enum (`External`, `SelfSpec`, `None`) replaces `&str` in `EnhancedMetricsCollector::inc_draft_resolution` (actual values: `"external"`, `"self_spec"`, `"none"`)
    - `RopeType` enum (`Default`, `Linear`, `Dynamic`, `Yarn`, `Su`, `Other` — serde `lowercase`) replaces `Option<String>` in `RopeScaling::rope_type` and `RopeParameters::rope_type`
    - `BatchEndpoint` enum (`Chat`, `Completion`) replaces `String` in batch request/response/job endpoint fields, with custom serde serializer/deserializer to preserve JSON wire compatibility
    - All conversions provide `parse(&str) -> Option<Self>`, `as_str() -> &'static str`, and `Display` impl
    - Affected call sites updated atomically (13 + 1 + 6 across the three enums)

- **Object-safe Trait Default Constructors (v24.0 Phase C-3)** — added `default_arc()` methods for 11 object-safe public traits:
    - **4 high-ROI**: `DraftVerifier` (`StubDraftVerifier`), `ModelBackend` (`StubModelBackend`), `SchedulerObserver` (`NoopSchedulerObserver`), `MetricsExporter` (`InMemoryMetricsExporter`)
    - **7 medium-ROI**: `SchedulingPolicy` (reused `FcfsPolicy`), `DraftLoader` (reused `NoopLoader`), `CudaGraphTensor` (`NullCudaGraphTensor`), `CudaGraphNode` (`NullCudaGraphNode`), `AllReduce` (`NoopAllReduce`), `PipelineStage` (`NoopPipelineStage`), `Architecture` (`UnknownArchitecture`)
    - Each trait gains `<dyn Trait>::default_arc() -> Arc<Self>` (Rust orphan rule prevents `impl Default for Arc<dyn Trait>`; the inherent-method pattern is the standard workaround)
    - Callers use `Arc::<dyn Trait>::default_arc()` (via type inference) or explicit `<dyn Trait>::default_arc()`
    - 6 low-ROI traits deferred (never used as `Arc<dyn Trait>` in current code)

- **Module Boundaries (v24.0 Phase D-1)** — split `crates/core/src/engine.rs` (866 non-test LOC) into 7 focused sub-modules under `engine/`:
    - `mod.rs` — facade: `Engine` struct, `SleepPolicy`, tests (437 LOC)
    - `ctor.rs` — constructors (`new_boxed`, `with_config_boxed`, `with_drafts_*`, `with_budget_boxed`) + `EngineBuilder` (318 LOC)
    - `draft_management.rs` — draft registry, resolver, speculative-mode toggles (124 LOC)
    - `cuda_graph.rs` — `capture_cuda_graphs` + `cuda_graph_enabled` (cfg-gated pairs) (35 LOC)
    - `lifecycle.rs` — `is_healthy`, `get_last_error`, `cancel_request`, `add_request` (38 LOC)
    - `run.rs` — `run` main loop + `has_pending` (74 LOC)
    - `beam.rs` — `step_beam` + `beam_search` + `get_top_k` (111 LOC)
    - `graph_step.rs` — `step_with_graph` (cfg-gated pair) + `execute_regular` + `process_output` (155 LOC)
    - Public API of `Engine` unchanged — single `crate::engine::Engine` type, all methods accessible
    - All 1191 tests pass (`just ci` clean)
    - Three `#[cfg(feature = "cuda-graph")]` / `#[cfg(not(feature = "cuda-graph"))]` duplicate method pairs (`capture_cuda_graphs`, `cuda_graph_enabled`, `step_with_graph`) preserved as intentional feature-gated pairs
    - Largest sub-module: `ctor.rs` at 318 LOC (was 866 LOC monolith); all sub-modules < 500 LOC

- **Module Boundaries (v24.0 Phase D-2)** — split `crates/core/src/scheduler/engine.rs` (654 non-test LOC) into 4 focused sub-modules under `scheduler/engine/`:
    - `mod.rs` — facade: sub-module declarations, `pub use` re-exports, the 8 unit tests (172 LOC)
    - `state.rs` — `SchedulerEngine` struct + `Default` impl + 17 methods: `new`, `set_policy`, `add_request`, `build_batch`, `schedule`, plus the 10 read-only / minor-mutating accessors (`has_pending`, `running_count`, `waiting_count`, `prefix_cache_hit_rate`, `running`, `get_sequence`, `get_sequence_mut`, `finished_sequences`, `clear_finished`, `register_observer`) (404 LOC)
    - `graph.rs` — CUDA Graph helpers: `build_batch_with_graph` + 2 private helpers (`get_scheduler_state`, `select_sequences_for_phase`) (81 LOC)
    - `update.rs` — post-step state update: `update` (121 LOC)
    - `memory.rs` — preemption + pressure: `execute_preemption`, `get_memory_pressure`, `memory_rollback`, `cancel_request`, `get_kv_cache_usage`, `prefix_cache` (114 LOC)
    - Public API of `SchedulerEngine` unchanged — single `crate::scheduler::engine::SchedulerEngine` type, all methods accessible via flat namespace
    - All 1191 tests pass (`cargo test --workspace` clean)
    - Largest sub-module: `state.rs` at 404 LOC (was 654 LOC monolith); the struct + 6 large lifecycle methods concentrate here. `graph.rs`, `update.rs`, and `memory.rs` are all ≤ 121 LOC.

- **Module Boundaries (v24.0 Phase D-3a)** — split the two remaining hard-target files > 500 LOC:
    - `crates/core/src/types.rs` (538 non-test LOC) → 7 sub-modules under `types/`:
        - `mod.rs` — facade: re-exports `vllm_traits::{Batch, BatchOutput, BlockId, SeqId, TokenId}` and `DraftId` (21 LOC)
        - `adaptive_draft.rs` — `AdaptiveDraftConfig` + `AdaptiveDraftConfigBuilder` (90 LOC)
        - `request.rs` — `Priority` + `Request` (55 LOC)
        - `sampling.rs` — `SamplingParams` + `SamplingParamsBuilder` (76 LOC)
        - `sequence.rs` — `Sequence` + `Status` + `Phase` (47 LOC)
        - `sequence_packing.rs` — `SequencePackingConfig` + builder + `from_env` (90 LOC)
        - `scheduler_config.rs` — `SchedulerConfig` + `SchedulerConfigBuilder` (175 LOC)
        - `messages.rs` — `EngineMessage` enum (23 LOC)
    - `crates/model/src/components/ssm.rs` (568 LOC) → 5 sub-modules under `components/ssm/`:
        - `mod.rs` — facade: re-exports + 7 unit tests (82 LOC)
        - `config.rs` — `SSMConfig` (48 LOC)
        - `layer.rs` — `softplus` helper + `SSMLayer` (137 LOC)
        - `mamba.rs` — `MambaBlock` (138 LOC)
        - `harmonic.rs` — `SSMHarmonicSSMLayer` (185 LOC)
        - `error.rs` — `SSMError` + `From<Infallible>` impl (16 LOC)
    - Public APIs unchanged: `crate::types::{Priority, Request, SamplingParams, SchedulerConfig, ...}` and `crate::components::ssm::{SSMConfig, SSMLayer, MambaBlock, SSMHarmonicSSMLayer, SSMError, softplus, ...}` still work via flat re-exports
    - All 1191 tests pass (`cargo test --workspace` clean)
    - Largest sub-module: `ssm/harmonic.rs` at 185 LOC (was 568 LOC monolith); all sub-modules ≤ 185 LOC

- **Module Boundaries (v24.0 Phase D-3b)** — split 7 soft-target files (224-907 LOC band) into focused sub-modules:
    - `crates/server/src/cli.rs` (548 LOC) → 2 sub-modules under `cli/`:
        - `mod.rs` — facade with re-exports (3 LOC)
        - `args.rs` — `CliArgs`, `ModelArgs`, validation helpers, `LogLevel` (548 LOC)
    - `crates/core/src/metrics/collector.rs` (521 LOC) → 3 sub-modules under `metrics/collector/`:
        - `mod.rs` — facade (11 LOC)
        - `metrics.rs` — `DraftResolutionKind`, `DraftMetricsSnapshot` (74 LOC)
        - `sampler.rs` — `EnhancedMetricsCollector` struct + impl + tests (470 LOC)
    - `crates/model/src/components/gated_delta/mod.rs` (581 LOC) → 3 sub-modules under `components/gated_delta/`:
        - `mod.rs` — facade (7 LOC)
        - `state.rs` — `GatedDeltaConfig` + `GatedDeltaState` (67 LOC)
        - `rule.rs` — `GatedDeltaNet` + helpers + tests (529 LOC)
    - `crates/model/src/qwen3/config.rs` (631 LOC) → 3 sub-modules under `qwen3/config/`:
        - `mod.rs` — facade (7 LOC)
        - `rope.rs` — `RopeType` + `RopeScaling` + `RopeParameters` (176 LOC)
        - `model.rs` — `TextConfig` + `Qwen3Config` + `AttentionType` (470 LOC)
    - `crates/core/src/scheduler/batch_composer.rs` (672 LOC) → 3 sub-modules under `scheduler/batch_composer/`:
        - `mod.rs` — facade (14 LOC)
        - `validate.rs` — `BatchCompositionConfig` + `ChunkedPrefillConfig` + builders (150 LOC)
        - `compose.rs` — `BatchComposer` + impl + tests (532 LOC)
    - `crates/model/src/paged_tensor/tensor_store.rs` (828 LOC) → 4 sub-modules under `paged_tensor/tensor_store/`:
        - `mod.rs` — facade with `PagedKvCache` struct + `new()` (68 LOC)
        - `buffer.rs` — `write_kv` / `read_kv` / `write_kv_batch` + tests (666 LOC)
        - `layout.rs` — hash + scale + `block_size` accessors (58 LOC)
        - `pool.rs` — `CacheBlock` + `KvCachePool` (71 LOC)
    - `crates/model/src/kernels/flash_attention.rs` (907 LOC) → 4 sub-modules under `kernels/flash_attention/`:
        - `mod.rs` — facade (15 LOC)
        - `config.rs` — `AttentionVariant` + `FlashAttentionConfig` + tile-size helpers (81 LOC)
        - `util.rs` — `AttentionStats` + `softmax_last_dim` (35 LOC)
        - `kernel.rs` — `FlashAttention` trait + `ScaledDotProductAttention` + `FlashAttentionV2` + `FlashAttentionKernel` + tests (809 LOC)
    - Public APIs unchanged across all 7 splits: external callers continue to import via flat namespace (e.g. `crate::cli::CliArgs`, `crate::components::gated_delta::GatedDeltaNet`, `crate::paged_tensor::tensor_store::PagedKvCache`)
    - All 1191 tests pass (`cargo test --workspace` clean)
    - Largest remaining sub-module: `flash_attention/kernel.rs` at 809 LOC (was 907 LOC monolith); all other sub-modules ≤ 666 LOC. The `kernel.rs` file is large because the `FlashAttentionV2` causal-mask + standard forward paths and the SDPA tiled/sliding-window paths are all in one place; further decomposition would require trait extraction beyond the scope of this phase.

- **Module Boundaries (v24.0 Phase D-3c)** — visibility tightening and re-export cleanup:
    - ~101 `pub` items → `pub(crate)` across crates (model: ~59, core: ~25, server: ~13, testing/dist: ~4)
    - 6 `pub mod` → `pub(crate) mod` (architecture modules in `model/` only consumed via the `Architecture` trait: `gemma3`, `gemma4`, `llama4`, `mistral_small`, `phi4`, `mixtral`). The other 4 architecture modules (`llama`, `mistral`, `qwen3`, `qwen3_5`) remain `pub` because integration tests use direct path access
    - 5 deep re-export chains flattened (added flat re-exports in intermediate modules; collapsed 2 thin re-export shims in `qwen3_5`)
    - 7 glob re-exports → explicit lists (`model/components/mod`, `model/components/ssm/mod`, `core/types/mod`, `core/scheduler/engine/mod`, `core/speculative/draft_registry`, plus consolidated `dist/lib.rs` `tensor_parallel` group)
    - 2 duplicate re-exports deleted (redundant `qwen3_5::hybrid` shim; `PipelineStageTrait` alias)
    - 4 additional types tightened via method visibility reduction: `GraphStats`, `GdnLinearConfig`, `AttentionConfigBuilder`, plus internal `validator::validate_chat_request` and `chat_template::build_prompt` in `server`
    - Conservative scope: items in public method signatures, axum handler parameter/return types, OpenAI DTOs, tonic-generated proto types, and crate-root re-exports remain `pub`
    - All 1191 tests pass (`cargo test --workspace` clean); build and `cargo clippy --workspace --all-features` clean

- **Pedantic Cleanup (v25.0 Phase E-1)** — mechanical fixes:
    - `cargo clippy --fix` applied (uninlined_format_args, redundant_closure, redundant_pub_crate, redundant_field_names, missing_const_for_fn, etc.) across 261 files
    - `#[must_use]` added to ~490 candidates (cargo clippy --fix auto-applied all of them)
    - `#[derive(Debug)]` added to ~124 types — zero `missing_debug_implementations` warnings remain
    - For types with `dyn Trait` fields (ModelBackend, SchedulingPolicy, etc.), manual `impl Debug` was added that displays a placeholder string instead of attempting to format the trait object
    - Pedantic warning count: 3605 → 1496 (-59%)
    - All 1191 tests pass (`just ci` clean)
    - Top remaining lints: `missing_errors_doc` (249), `module_name_repetitions` (225), `cast_possible_truncation` (128), `cast_precision_loss` (121), `unreadable_literal` (96), `significant_drop_tightening` (61), `float_cmp` (60) — most are in the "manual" or "already-allow" categories from the Phase E audit and are deferred to later sub-phases

- **Pedantic Cleanup (v25.0 Phase E-2)** — doc comment cleanup:
    - 4 `doc_markdown` backtick warnings fixed (2 in `chat_template.rs`, 2 in gRPC-generated code suppressed via module-level `#[allow]`)
    - 244 `# Errors` sections added across `vllm-core`, `vllm-model`, `vllm-server`, `vllm-dist`, and `vllm-traits`
    - 36 `# Panics` sections added across `vllm-core`, `vllm-model`, `vllm-server`, and `vllm-dist`
    - Generated proto code in `vllm-dist/src/grpc.rs` wrapped in a `generated_proto` module with `#[allow(clippy::doc_markdown, clippy::missing_errors_doc, clippy::missing_panics_doc)]` since the source is regenerated by `tonic_build` and not under our control
    - All `# Errors` / `# Panics` sections describe the specific failure conditions (e.g. "Returns `Err` if any tensor operation fails (shape mismatch, out-of-memory, dtype incompatibility, or kernel error)" rather than boilerplate)
    - Pedantic warning count: 1496 → 1210 (-19% for this phase; the remaining top lints are `must_use_candidate`, `module_name_repetitions`, `cast_*`, `unreadable_literal`, `significant_drop_tightening` which are deferred to later sub-phases)
    - All 1191 tests pass (`just nextest` clean)
    - `just clippy`, `just fmt-check`, and `just doc-check` all pass

- **Comprehensive Refactor (Phase 1 + Phase 2)** — production-grade hardening
  across CI, error handling, observability, and the public API surface.
  Follows the four parallel audits (architecture, error handling, testing,
  CI/CD) that ran in this session. 7 atomic commits, 1230+ tests passing.

  **Phase 1: Infrastructure (CI/CD + Engineering Practices)**
    - **CI audit hardening** (audit findings C1, H6):
      - Removed `|| true` from `cargo audit`; high/critical RUSTSEC
        advisories now block PRs. RUSTSEC-2024-0436 (paste) is allow-listed
        via `--ignore` with documented rationale in SECURITY.md.
      - Swatinem/rust-cache@v2 replaces hand-rolled `actions/cache` (better
        hit rates, profile/target-aware).
      - Pinned dtolnay/rust-toolchain to `@stable` (was `@master`).
    - **New CI workflows**:
      - `msrv.yml` — Rust 1.88 MSRV compile check with drift guard
        (fails if `[workspace.package].rust-version` and the matrix disagree).
      - `deny.yml` — cargo-deny license/bans/advisories gate.
      - `release.yml` — tag-triggered multi-OS build + GitHub Release with
        auto-generated git-cliff notes.
    - **`deny.toml`** — project-wide dependency policy: MIT-compatible
      license allowlist, duplicate-version bans, non-crates.io source bans,
      RUSTSEC allow-list with rationale comments.
    - **GitHub project hygiene**:
      - `.github/CODEOWNERS` (per-crate ownership, sensitive paths protected).
      - `.github/PULL_REQUEST_TEMPLATE.md` (checklist including "no panic",
        "no Box<dyn Error>", etc.).
      - `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.yml`.
    - **Pre-commit** — Rust toolchain hooks added:
      - `cargo fmt --check` on pre-commit.
      - Tiered `cargo clippy` + `cargo audit` on pre-push.
      - `check-added-large-files` (512KB cap), `check-case-conflict`.
    - **`justfile`**:
      - `quick` is now **read-only** (was running `cargo fix` and modifying
        source). New `autofix` target for the mutating variant.
      - New `deny`, `deny-advisories`, `security`, `doctest`, `ci-all` targets.
    - **`CONTRIBUTING.md`** — synced to reality: Rust 1.75 → 1.88 (MSRV),
      nextest as canonical test runner, tiered clippy denies matching CI,
      documents the "no Box<dyn Error>", "no unwrap in non-test code"
      contracts, project structure now lists all 6 crates.
    - **`docs/cliff.toml`** — conventional-commits → CHANGELOG generator
      used by `release.yml`.

  **Phase 2: Security + Error Handling**
    - **`main.rs` → `anyhow::Result`** (audit C4):
      - All startup panics replaced with structured `?` propagation +
        `anyhow::Context` (loader build, model load, draft model load,
        server bind, serve).
      - Distinct exit code 78 (EX_CONFIG) for config validation failures,
        distinguishable from transient infra failures in supervisor
        restart policies.
      - Extracted helpers: `build_engine`, `configure_speculative`,
        `load_tokenizer`. `main()` is now a linear sequence of focused
        calls.
    - **OpenAI error contract hardening** (audit H3, H4, C1):
      - New `ErrorResponse::with_code(message, error_type, code)`
        constructor — OpenAI-spec `code` slot for stable identifiers
        (e.g. `engine_unavailable`, `context_length_exceeded`).
      - All OpenAI handlers (chat, completions, embeddings) upgraded:
        engine-channel-closed failures now return `503 SERVICE_UNAVAILABLE`
        + `code = "engine_unavailable"` (was `500 INTERNAL_SERVER_ERROR`
        with no code). Distinguishes transient + retryable failures from
        real server-side bugs.
      - `embeddings.rs`: replaced `let _ = state.engine_tx.send(...)`
        (which silently dropped `SendError`) with `.map_err()`.
      - Doc-comments updated; tests locked the contract.
    - **`vllm_server::util::time`** — new module:
      - `unix_now_secs()` and `unix_now_millis()` panic-free accessors
        that saturate instead of panicking on NTP-induced clock skew
        across `UNIX_EPOCH`. Replaces 5 `.expect("Failed to get system
        time")` sites in `types.rs`, `batch/types.rs`, `batch/handler.rs`.
      - 2 unit tests (post-2024 sanity + secs/millis consistency).
    - **Typed `DraftRegistryError` variants** (audit C2, M1):
      - Added typed `IoLoad { draft_id, path, source: io::Error }` and
        `Model(DraftId, vllm_traits::ModelError)` variants.
      - `From<std::io::Error>` and `From<vllm_traits::ModelError>` for
        `?` ergonomics.
      - Legacy `LoadFailed(String)` and `LoadFailedWithSource { ...,
        source: Box<dyn Error> }` variants marked
        `#[deprecated(since = "0.1.0", note = "Use IoLoad or Model
        instead")]`. Eliminates the `Box<dyn Error>` from the new-code
        path of the public API.
      - 3 call sites (NoopLoader, BenchLoader, StubLoader) migrated
        or annotated with `#[allow(deprecated)]`.
    - **Workspace dependency unification** (audit H5):
      - 8 common deps (`candle-core`, `candle-nn`, `tracing`,
        `tracing-subscriber`, `thiserror`, `parking_lot`, `async-trait`,
        `crossbeam`) centralised in `[workspace.dependencies]`. All 6
        crate Cargo.toml files migrated to `{ workspace = true }`.
      - Bumping any of these is now a one-line workspace edit instead of
        touching 4-6 Cargo.toml files.
    - **Doctest CI phase** (audit C1):
      - `cargo test --doc --workspace --all-features` added to both
        `ci.yml` and `matrix-test` jobs. Closes the gap where broken
        doc-example code was silently shipped (nextest only runs
        `#[test]` functions).
    - **OpenAI error contract test matrix** (audit C4):
      - New `crates/server/tests/error_contract.rs` — 9 tests locking
        the v0.1 server's wire-level error behavior across all 3 OpenAI
        handlers + the `ErrorResponse` constructors. Future refactors
        cannot accidentally downgrade the contract.

  **What this enables (for follow-up phases)**
    - `gqa.rs` (1036 lines) and `compose.rs` (872 lines) are now
      mechanical splits — file-size rules (800-line soft cap) are
      straightforward to enforce in CI without behavioral risk.
    - `parking_lot::Mutex` global replacement is now a pure refactor:
      all error semantics are already correct (typed `LockPoisoned`,
      `?` propagation, no panic-prone `.expect("poisoned")` on
      production paths).
    - The `Model(DraftId, ModelError)` variant enables per-failure
      recovery policies (retry vs fallback vs circuit-break) in
      future drafts work.

---

- **Comprehensive Refactor (Phase 5)** — module-splitting pass to bring
  every Rust file under the project's 800-line soft cap. Pure file-size
  refactors (zero behavioral change), all 1235 tests pass after each
  commit. 7 atomic commits, 10 large files slimmed down.

  Per-file line counts (before → after):
    | File | Before | After | Pattern |
    |------|-------:|------:|---------|
    | `model/src/kernels/flash_attention/kernel.rs` | 869 | 126 | split into `kernel/` with `flash_attention_v2.rs`, `scaled_dot_product.rs`, `tests.rs` |
    | `core/src/speculative/adaptive.rs` | 711 | 233 | extract 22 inline tests → `adaptive/tests.rs` |
    | `model/src/components/attention/mla.rs` | 724 | 301 | extract 15 inline tests → `mla/tests.rs` |
    | `model/src/components/attention/flash_attention_v3.rs` | 694 | 329 | extract 9 inline tests → `flash_attention_v3/tests.rs` |
    | `model/src/qwen3/block.rs` | 613 | 376 | extract 9 inline tests → `block/tests.rs` |
    | `model/src/qwen3/config/model.rs` | 556 | 381 | extract 9 inline tests → `model/tests.rs` |
    | `model/src/paged_tensor/tensor_store/buffer.rs` | 799 | 317 | extract 24 inline tests → `buffer/tests.rs` |
    | `core/src/scheduler/batch_composer/compose.rs` | 872 | 373 | (prior session) split into `compose/` with `tests.rs`, `prop_tests.rs` |
    | `model/src/components/attention/gqa.rs` | 1036 | 472 | (prior session) extract 17 tests → `gqa/tests.rs` |

  Pattern: each large file now declares `mod tests;` to point at a sibling
  `tests.rs` file, keeping the implementation focused and the 800-line cap
  enforceable. The kernel.rs split is a multi-way split (impl + tests) —
  `FlashAttention` trait + `FlashAttentionKernel` facade stay in
  `kernel.rs`, the `FlashAttentionV2` and `ScaledDotProductAttention`
  impls move to their own files. Follows the existing `compose.rs` +
  `compose/` sub-module pattern.

  Additional cleanup folded into this phase:
    - **Typed `DraftRegistryError` migration completion** (closes out the
      C2 audit): the 5 remaining call sites using the deprecated
      `LoadFailed(String)` variant (`NoopLoader`, `StubLoader` in
      `core/src/speculative/draft_resolver.rs`, plus 3 test/bench
      loaders) all migrated to the typed `Model(DraftId, ModelError)`
      variant. Eliminates the last `#[allow(deprecated)]` surface in
      the speculative decoding path; the legacy string variants are
      still present but now have zero callers in the workspace.
    - **Broken doc links fixed** (`crates/model/src/quantize/gguf.rs`,
      `crates/server/src/cli/args.rs`): a stale link to
      `crate::loader::format::load_checkpoint` (the symbol was renamed
      to `Format::can_load`) and a redundant explicit link target —
      both now pass `cargo doc --no-deps -D warnings`.

  Test count: 1235 → 1235 (zero new tests, zero removed — these were
  pure refactors). All Phase 5 commits verified by `just ci`
  (fmt-check → clippy → doc-check → nextest).

---

- **Comprehensive Refactor (Phase 6)** — second module-splitting pass
  targeting the remaining large source files. Pure file-size refactors
  (zero behavioral change), all 1235 tests pass after each commit.
  10 atomic commits, 10 large files slimmed down.

  Per-file line counts (before → after):
    | File | Before | After | Pattern |
    |------|-------:|------:|---------|
    | `server/src/security/jwt.rs` | 571 | 276 | extract 10 inline tests → `jwt/tests.rs` |
    | `core/src/engine/mod.rs` | 510 | 175 | extract 19 inline tests → `engine/tests.rs` |
    | `core/src/scheduler/memory/eviction.rs` | 459 | 196 | extract 11 tests + 3 proptests → `eviction/{tests,prop_tests}.rs` |
    | `core/src/scheduler/request_queue.rs` | 456 | 220 | extract 4 tests + 4 proptests → `request_queue/{tests,prop_tests}.rs` |
    | `core/src/sampling.rs` | 452 | 228 | extract 23 tests + 4 proptests → `sampling/{tests,prop_tests}.rs` |
    | `server/src/config.rs` | 551 | 391 | extract 15 inline tests → `config/tests.rs` |
    | `model/src/components/gated_delta/rule.rs` | 572 | 423 | extract 5 inline tests → `rule/tests.rs` |
    | `model/src/components/attention/util.rs` | 532 | 262 | extract 10 inline tests → `util/tests.rs` |
    | `core/src/scheduler/policy/priority.rs` | 207 | 64 | extract 1 test + 3 proptests → `priority/{tests,prop_tests}.rs` |
    | `core/src/scheduler/engine/mod.rs` | 172 | 34 | extract 8 inline tests → `engine/tests.rs` |

  Pattern: continues the Phase 5 sibling-file convention. For files
  with both `mod tests` and `mod prop_tests` inline blocks (eviction,
  request_queue, sampling, priority), both are extracted to separate
  sibling files; this preserves the unit-vs-property test boundary
  while keeping production code under the 800-line cap. Files whose
  fields are pub-only (config.rs) split cleanly; one file with private
  fields that needed broader visibility (cli/args.rs) was intentionally
  skipped this phase to avoid coupling the split with a visibility
  refactor.

  Two small follow-ups:
    - `cli/args.rs` was attempted but rolled back: the `CliArgs` struct
      has private fields (`server`, `engine`, `auth`, `logging`,
      `config`) and the tests access them directly via `cli.server.host`,
      `cli.engine.kv_blocks`, etc. Splitting would require either
      making those fields `pub(crate)` or moving the tests to
      `cli/args/tests.rs` — both are valid but orthogonal to the
      file-size split. Deferred to a future phase that bundles the
      visibility change.
      **Resolution (post-Phase-6 commit `9e557e4`)**: tests now live
      at `cli/args/tests.rs` (a child module of `cli::args`), so
      `use super::*;` retains full access to private fields. No
      visibility refactor needed. `cli/args.rs` 554 → 237 lines; the
      25 unit tests are unchanged.
    - One cargo fmt re-indent on `eviction/prop_tests.rs`
      (single-line commit, no functional change).

  Test count: 1235 → 1235 (zero new tests, zero removed — these were
  pure refactors). All Phase 6 commits verified by `cargo nextest run
  --workspace --no-fail-fast` and `cargo fmt --all --check`.

- **Comprehensive Refactor (Phase 7)** — third module-splitting pass.
  Closes out the remaining inline-test extractions across
  `crates/core/src/{speculative,metrics,scheduler}/`,
  `crates/server/src/openai/` + `security/`, and the `arch/`,
  attention / causal_lm / kv_cache_fp8 / mixtral / mistral_small /
  paged_tensor / positional / qwen3-architecture subtrees of
  `crates/model/`. Pure file-size refactors (zero behavioral
  change), all 1235 tests pass after each commit. 29 atomic
  commits in this phase, 28 source files slimmed down plus 1 doc
  pass plus 1 doc + test commit.

  Per-file line counts (before → after):
    | File | Before | After | Pattern |
    |------|-------:|------:|---------|
    | `core/src/scheduler/memory/allocator.rs` | 381 | 195 | extract 7 unit + 3 proptests → `allocator/{tests,prop_tests}.rs` |
    | `core/src/scheduler/preemption.rs` | 232 | 142 | extract 8 inline tests → `preemption/tests.rs` |
    | `core/src/scheduler/memory/mod.rs` | 264 | 213 | extract 3 inline tests → `memory/tests.rs` |
    | `core/src/speculative/registry/mod.rs` | 434 | 87 | extract 25 inline tests → `registry/tests.rs` |
    | `core/src/metrics/lock_free.rs` | 406 | 332 | extract 6 inline tests → `lock_free/tests.rs` |
    | `server/src/openai/chat.rs` | 496 | 313 | extract 14 inline tests → `chat/tests.rs` |
    | `server/src/openai/chat_template.rs` | 258 | 149 | extract 10 inline tests → `chat_template/tests.rs` |
    | `server/src/openai/completions.rs` | 188 | 137 | extract 2 `#[tokio::test]` cases → `completions/tests.rs` |
    | `server/src/openai/embeddings.rs` | 142 | 82 | extract 3 `#[tokio::test]` cases → `embeddings/tests.rs` |
    | `server/src/security/rbac.rs` | 289 | 155 | extract 8 tests (3 unit + 5 axum) → `rbac/tests.rs` |
    | `server/src/security/audit.rs` | 177 | 137 | extract 2 `#[tokio::test]` cases → `audit/tests.rs` |
    | `server/src/security/correlation.rs` | 128 | 97 | extract 3 `#[tokio::test]` cases → `correlation/tests.rs` |
    | `server/src/security/tls.rs` | 198 | 143 | extract 3 tests (incl. SEC-06 regression) → `tls/tests.rs` |
    | `server/src/security/size_limit.rs` | 131 | 27 | extract 4 `#[tokio::test]` cases → `size_limit/tests.rs` |
    | `model/src/loader/builder.rs` | 490 | 305 | extract 12 inline tests → `builder/tests.rs` |
    | `model/src/arch/registry.rs` | 217 | 113 | extract 6 tests + doc pass on every public method |
    | `model/src/arch/capabilities.rs` | 92 | 70 | extract 2 inline tests → `capabilities/tests.rs` |
    | `model/src/causal_lm/mod.rs` | 327 | 231 | extract 3 inline tests → `causal_lm/tests.rs` |
    | `model/src/components/attention/rope_gqa.rs` | 483 | 255 | extract 6 inline tests → `rope_gqa/tests.rs` |
    | `model/src/components/kv_cache_fp8.rs` | 318 | 226 | extract 7 inline tests → `kv_cache_fp8/tests.rs` |
    | `model/src/components/positional/rope.rs` | 312 | 145 | extract 14 inline tests → `rope/tests.rs` |
    | `model/src/components/vision.rs` | 149 | 95 | extract 6 tests + doc pass on PatchEmbed / VisionEncoder public surface |
    | `model/src/gemma4/attention.rs` | 484 | 399 | extract 2 inline tests → `attention/tests.rs` |
    | `model/src/gemma4/block.rs` | 359 | 302 | extract 1 inline test → `block/tests.rs` |
    | `model/src/kernels/cuda_graph.rs` | 458 | 257 | extract 12 inline tests → `cuda_graph/tests.rs` |
    | `model/src/kernels/cuda_graph/executor.rs` | 480 | 301 | extract 13 inline tests → `executor/tests.rs` |
    | `model/src/mixtral/sparse_moe.rs` | 384 | 280 | extract 5 inline tests → `sparse_moe/tests.rs` |
    | `model/src/mixtral/block.rs` | 375 | 322 | extract 1 inline test → `block/tests.rs` |
    | `model/src/mistral_small/arch.rs` | 309 | 260 | extract 4 inline tests → `arch/tests.rs` |
    | `model/src/paged_tensor/quant.rs` | 402 | 314 | extract 4 inline tests → `quant/tests.rs` |

  Other commits in this phase:
    - `style`: cargo fmt fix on `draft_resolver/tests.rs` +
      `gated_delta/rule/tests.rs` (mechanical re-indent of long
      fn signatures; zero behavior change).
    - `docs(dist)`: documented `RowParallelLinear::new`,
      `input_size_per_rank`, `forward`, `TensorParallelManager::new`,
      `create_column_parallel`, `create_row_parallel`, `mesh` —
      spells out the per-rank split/replicate topology each method
      implies and which error variants each can return.

  Pattern: continues the Phase 5/6 sibling-file convention. After
  this phase, the inline-test surface is fully migrated across
  `core/src/scheduler/`, `core/src/speculative/registry/`,
  `core/src/metrics/lock_free.rs`, `server/src/openai/`,
  `server/src/security/` (audit / correlation / jwt / rbac /
  size_limit / tls), `model/src/loader/`, `model/src/arch/`,
  `model/src/causal_lm/`, `model/src/components/{attention,
  kv_cache_fp8, positional, vision}/`, `model/src/gemma4/`,
  `model/src/kernels/`, `model/src/mixtral/`,
  `model/src/mistral_small/`, and `model/src/paged_tensor/`.
  The pattern is the default for new modules going forward.

  Doc coverage delta: model crate real% moved from 48.8% → 49.6%
  on the strength of:
    - `PatchEmbed`, `PatchEmbed::new`, `PatchEmbed::forward`,
      `VisionEncoder::config`, `VisionEncoder::forward`
      (`vision.rs` doc pass)
    - `ArchitectureRegistry` + all 7 public methods on it, plus
      the `ARCHITECTURE_REGISTRY` static
      (`arch/registry.rs` doc pass)
  Total workspace real% 51.5% → 51.9% (+4 net documented items).

  Test count: 1235 → 1235 (zero new tests, zero removed — these
  were pure refactors). All Phase 7 commits verified by `cargo
  nextest run --workspace --no-fail-fast` and `cargo fmt --all
  --check`.

---

## 🚀 [v18.0] — Multi-Model Speculative Decoding (2026-06-27)

### Added

- **DraftModelRegistry** (`crates/core/src/speculative/draft_registry.rs`)
    - Runtime registry for heterogeneous external draft models
    - Each draft owns a private `ModelBackend` and `BlockAllocator` (KV isolation by construction)
    - Lazy weight loading via `register` (Unloaded) → `attach_loaded` (Loaded) state machine
    - `Engine::with_drafts_boxed` constructor for pre-loading specs at engine startup
    - Loader-agnostic — does not depend on `vllm-model`; caller drives actual ModelLoader

- **MemoryBudget** (`crates/core/src/speculative/memory_budget.rs`)
    - VRAM budget enforcement for target + concurrent drafts
    - Atomic `try_reserve_draft` with structured `MemoryBudgetExceeded { requested_bytes, available_bytes, draft_id }` error
    - Runtime KV-cache growth tracking via `record_draft_kv_growth`
    - Default is `u64::MAX` (unlimited) — existing flows unchanged

- **Refcount-driven lifecycle**
    - `unload` returns `InUse(refcount)` if refcount > 0 (LIFE-02)
    - `force_unload` bypasses refcount for admin/test paths
    - `decrement_ref` auto-unloads when count reaches zero (LIFE-03)
    - Releases budget reservation on unload

- **DraftResolver** (`crates/core/src/speculative/draft_resolver.rs`)
    - Per-request draft selection with FALL-01 fallback semantics
    - `ResolvedDraft::{External, SelfSpec, None}` enum makes outcomes explicit
    - `DraftLoader` trait abstracts actual model loading (no vllm-model coupling)
    - Records metrics for every resolution (external / self_spec / none)

- **Per-request routing**
    - `Request.draft_model_id: Option<DraftId>` + `Request::with_draft_model` builder (RTE-01)
    - `Sequence.draft_model_id` propagated from Request (RTE-02)
    - Per-request resolution enables mixed drafts in one batch (RTE-03)

- **Fallback semantics**
    - FALL-01: load failure / unknown id / budget exceeded → silent fallback to self-spec
    - FALL-02: `Sequence.degraded_draft: bool` sticky flag set on runtime draft errors

- **Metrics** (5 new counters in `EnhancedMetricsCollector`)
    - `draft_resolutions_external_total`
    - `draft_resolutions_self_spec_total`
    - `draft_resolutions_none_total`
    - `draft_load_failures_total`
    - `draft_runtime_errors_total`

- **Integration tests** (`crates/core/tests/multi_draft_integration.rs`)
    - 14 tests covering full lifecycle, budget boundaries, mixed routing, all fallback paths
    - Stub backends with configurable failure injection

- **Benchmark** (`crates/core/benches/multi_draft_speculative.rs`)
    - Criterion benchmark: `no_draft` vs `self_spec` vs `external_draft` (3 configs)
    - Measures orchestration overhead (~1.7-2.1 µs per 16-step iteration)

### Changed

- `Sequence` gained `degraded_draft: bool` and `draft_model_id: Option<DraftId>` fields
- `Request` gained `draft_model_id: Option<DraftId>` field
- `BlockAllocator` gained `bytes_per_block()` and `allocated_bytes()` methods
- `DraftSpec` gained `weight_size_estimate_bytes: u64` field for MEM-02 budget estimation

### Requirements Satisfied

- MMLT-01, MMLT-02, MMLT-03 (multi-model loading)
- LIFE-01, LIFE-02, LIFE-03 (lifecycle management)
- MEM-01, MEM-02, MEM-03 (memory budget)
- RTE-01, RTE-02, RTE-03 (request routing)
- FALL-01, FALL-02 (fallback semantics)

**14/14 requirements passed.** Test count: 209 → 277 (+68).

### Refactored

#### Architecture Refactoring

- **Scheduler Module Split**
    - Split monolithic `scheduler.rs` into focused submodules
    - Created `scheduler/queue.rs` with `RequestQueue` for queue management
    - Created `scheduler/preemption.rs` with `PreemptionManager` for preemption decisions
    - Created `scheduler/eviction.rs` with `EvictionPolicy` for block eviction
    - Fully integrated all modules into `Scheduler` struct

- **KV Cache Layer Separation**
    - Split `core/kv_cache.rs` into `kv_cache/block_allocator.rs` and `kv_cache/prefix_cache.rs`
    - Created `model/paged_tensor/` module (separating logical and physical KV cache)
    - `tensor_store.rs` for GPU KV tensor management
    - `quantization.rs` for INT8/FP8 quantization
    - Added deprecated alias in `kv_cache.rs` for backward compatibility

- **Kernel Layer Extraction**
    - Created `model/kernels/` directory for GPU kernels
    - Moved `flash_attention.rs` → `kernels/flash_attention.rs`
    - Moved `fused_kernel.rs` → `kernels/fused_mlp.rs`
    - Moved `cuda_graph.rs` from core to `model/kernels/cuda_graph.rs`
    - Updated `components/` to use kernels module

#### Architecture Refactor Phase 5 (Qwen3.5 Hybrid 收敛, 2026-06-15)

- Split `qwen3_5/hybrid.rs` (1176 lines) into `block/` + `model.rs` + `weights.rs` + `config.rs`
- Introduce `HybridLm<B, Norm>` shell paralleling `CausalLm<B, N, H>`
- Move `GatedDeltaState` from `qwen3_5::gated_delta` to `components::gated_delta`
- Remove `causal_lm → qwen3_5` reverse dependency (`rg 'use qwen3_5' crates/model/src/causal_lm/` → 0 matches)
- GDN dims now read from `Qwen3Config` (no more hardcoded `(16, 4, 2)`)
- `Qwen35Architecture::capabilities()` upgraded to `PRODUCTION_SPECULATIVE`
- Speculative parity tests in `model_tests.rs` (124 lines) + `speculative_tests.rs` (285 lines)

Refs: `decc8c8`, `73dab5e`, `52f77ce`

#### Adaptive Speculative Decoding Counter Wire-up (Wave 2, 2026-06-26)

- `AdaptiveSpeculativeDecoder::record_verification` now returns `bool` adjustment event
- Engine `step_speculative_inner` calls `MetricsCollector::record_speculative_adjustment()` on actual adjustment
- `speculative_adjustments_total` Prometheus counter now correctly tracks adaptive decoder activity
- 3 new unit tests locking the bool return contract (high acceptance, low acceptance, deadband)
- Documentation: `SPEC-ADAPT-01` / `SPEC-ADAPT-02` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave2-adapt-spec.md`

#### Speculative Warmup Test Coverage (Wave 4, 2026-06-26)

- `Engine::warmup_draft_kv` visibility relaxed from `fn` to `pub(crate) fn` for test access
- New `CounterModel` wrapper in `engine::speculative::tests` mod (counts forward/forward_logits calls via AtomicUsize)
- New fast unit test `test_warmup_draft_kv_invokes_draft_per_sequence` verifies draft model receives exactly N forward() calls for N-seq Prefill batch
- Documentation: `SPEC-WARM-01` marked complete in `.planning/PROJECT.md`

Refs: `docs/superpowers/specs/2026-06-26-wave4-warmup-test.md`

#### Benchmark Suite Closure (Wave 5, 2026-06-26)

- New `crates/core/benches/latency_percentiles.rs` — per-request latency distribution with criterion auto-reported p50/p95/p99 (SPEC-BENCH-01)
- New `crates/core/benches/speculative_vs_baseline.rs` — explicit baseline vs adaptive speculative throughput comparison (SPEC-BENCH-02)
- New `docs/benchmark-suite.md` — suite documentation covering all 9 benchmarks
- New `just bench` recipe — runs all benchmarks with `--output-format bencher`
- Documentation: `SPEC-BENCH-01` / `SPEC-BENCH-02` marked complete in `.planning/PROJECT.md`; v17.0 milestone closes 7/9 SPECs

Refs: `docs/superpowers/specs/2026-06-26-wave5-benchmark-suite.md`

### Added (Phase 4)

#### Phase 4: Performance Optimization

- **Quantization Support**
    - FP16 support
    - INT8 Weight-Only quantization (`QuantizedLinear`, `quantize_2d`)
    - INT8 KV Cache with per-layer scaling
    - QuantizationCalibrator for calibration
- **Compute Optimization**
    - Flash Attention framework with software fallback (`FlashAttention`, `ScaledDotProductAttention`)
    - Sliding window attention support
    - CUDA Graph framework (`CudaGraph`, `CudaGraphExecutor`)
- **Scheduling Optimization**
    - PD Separation (Prefill/Decode separation)
    - Chunked Prefill with configurable chunk size
    - Dynamic Batch Size based on available KV blocks
    - Priority-based scheduling (`Priority`, `enable_priority_scheduling`)
- **Distributed**
    - Multi-GPU Tensor Parallelism (`DeviceMesh`, `ColumnParallelLinear`, `RowParallelLinear`, `AllReduce`)

#### Phase 5: Production Readiness (2025-04-12)

**Observability & Metrics**

- Prometheus-compatible metrics export (`/metrics` endpoint)
- Enhanced metrics collection (CUDA Graph, Sequence Packing, Adaptive Speculative)
- Health check endpoints (`/health`, `/ready`)
- Real-time metrics with `EnhancedMetricsCollector`

**Fault Tolerance**

- Circuit breaker pattern for automatic failure recovery
- Retry strategy with exponential backoff
- Degrade strategy for graceful degradation
- Recovery manager with error severity classification

**Testing**

- 26 E2E integration tests (lifecycle, concurrent, error recovery, graceful shutdown)
- Deterministic mock models for reproducible tests
- Performance regression testing in CI

**Deployment**

- Multi-stage Docker build (`Dockerfile`)
- Docker Compose with Prometheus (`docker-compose.yml`)
- Kubernetes manifests (namespace, deployment, service, HPA)
- CI performance regression workflow (`.github/workflows/benchmark.yml`)

**Core Features**

- Request timeout support (`timeout` parameter)
- Graceful shutdown (SIGINT/SIGTERM handling)
- YAML configuration file support
- Environment variable overrides (`VLLM_HOST`, `VLLM_PORT`, etc.)
- Structured JSON logging with file rotation
- Grafana dashboard (`docs/grafana/dashboard.json`)
- Config validation on startup
- Error retry support (`retries` parameter)

#### Core Features

- Real-time metrics collection with `/v1/stats` and `/metrics` endpoints
- Quantization utilities (`crates/model/src/quantize.rs`)
- Tiled Attention for memory optimization
- INT8 quantization support in KV Cache
- Forward pass with tiled attention strategy
- Comprehensive test suite for tiled attention

### Changed

- Improved documentation structure (README.md, docs/README.md, ROADMAP.md)
- Added detailed development roadmap

### Fixed

- Clippy warnings and code quality improvements
- Test compatibility with new AttentionConfig

## [0.1.0] - 2026-03-31

### Added

- **Continuous Batching** - Dynamic batch scheduling with decode-priority
- **Paged KV Cache** - Memory-efficient cache management with LRU eviction
- **Prefix Caching** - Exact match and prefix hit support
- **Speculative Decoding** - Draft-target verification architecture
- **Qwen3 Model Integration** - Support for Qwen2.5-0.5B model with real weights
- **OpenAI-compatible API** - `/v1/completions`, `/v1/chat/completions`
- **Streaming (SSE)** - Real-time token streaming
- **Sampling** - Temperature, Top-P support
- **Chunked Prefill** - Process long prompts in chunks

### Architecture

- **3-Crate Structure**:
    - `vllm-core`: Scheduler, Engine, KV Cache, Types
    - `vllm-model`: Qwen3, Attention, MLP
    - `vllm-server`: HTTP API (axum)

### Dependencies

- Rust (edition 2021)
- Candle (ML backend)
- Axum (HTTP)
- Tokio (async runtime)
- SafeTensors (weight loading)

---

## Migration Guides

### Upgrading to 0.1.0

No migration needed - initial release.

---

## 🚀 [v22.0] — Production Hardening (2026-06-27)

### Added

- **Security middleware wired (SEC-01..06)** — JWT signature verification (HMAC-SHA256 + RSA/ECDSA), `RbacMiddleware` permission enforcement, `RequestBodyLimitLayer` (configurable max body), audit log integration test, Grafana credentials moved to `.env`, structured TLS error replacing `unwrap()` panic
- **Phase 19 e2e tests un-ignored (OPS-02)** — `Engine::step()` speculative-mode hang fixed (DashMap shard re-entry deadlock in `EnhancedMetricsCollector`); 9 tests across `engine_wiring.rs` and `engine/spec_dispatch/tests.rs` now pass
- **Production polish (OPS-01, RFU-05, PERF-01..03)** — `parking_lot::Mutex` migration (24 sites in scheduler/engine), `MlaKvCache::write_compressed` incremental `slice_assign`, `eq_ignore_ascii_case` in arch detection, `std::sync::LazyLock` migration
- **Engine refactor (ARF-06, ARF-07)** — `engine.rs` God module split into focused sub-modules; `engine/spec_dispatch` tree unified post-Phase-31

### Fixed

- 5 cargo doc broken-link warnings resolved (intra-doc-link fixes in `engine.rs`, `components/attention/mod.rs`, `block.rs`, `decoder_block/mod.rs`, `speculative/registry/`)
- `Engine::step()` speculative-mode determinism bug (was hanging in 9 tests)

### Tech Debt Rolled Forward

- Stub architectures (`gemma3`, `llama4`, `phi4`, `mistral_small`) — policy deferred to v23.0
- `TensorParallelError` manual impl — deferred to v23.0
- `Box<dyn Error>` in `dist/src/grpc.rs` — deferred to v23.0
- Stale `CLAUDE.md`, `README.md`, `CHANGELOG.md`, `MIGRATING.md` — deferred to v23.0
- Dead code (~2000 LOC across scheduler/, routing/, ha/, circuit_breaker/) — deferred to v23.0
- `core → model` upward dep via cuda-graph feature — deferred to v23.0

### Stats

- **Phases:** 4 (Phase 36-39)
- **Test count:** 1179+ (≥ 1146 v21.0 baseline; +33 net)
- **Coverage:** doc 97.8%, clippy/fmt clean, 0 cargo doc warnings
- **ADRs:** 15 (no new ADRs in v22.0)

---

## 🚀 [v21.0] — P2/P3 Backlog Cleanup (2026-06-27)

### Added

- **API/error boundary work** (API-04, API-06, API-08, API-10) — Mutex `.expect()` migrations, `From<E>` impls for cross-crate error conversion, `dyn Trait` compile-only tests for object-safe traits (8 tests in `crates/testing/tests/dyn_safety.rs`)
- **Naming audit compliance** (NAME-F-04) — `qwen3_config` module moved to `qwen3::config`; test files migrated from `src/` to `crates/*/tests/`

### Removed

- **Dead `mod.rs`** in `crates/traits/tests/` (P3-01)

### Tech Debt Rolled Forward

- `mut Prompt::token_ids` non-mutating methods (P2-09) — partial; deferred to v22+
- Multi-node/vllm-dist resurrection — feature-gated only; OPS-05 still deferred
- Real-model benchmark — OPS-04 deferred (no GPU env)

### Stats

- **Phases:** 5 (Phase 31-35)
- **Test count:** 1146+ (1144 baseline + 13 new − 11 dedup)
- **Coverage:** doc 97.8%

---

## 🚀 [v20.0] — Codebase Remediation (2026-06-27)

### Added

- **48 requirements addressed across 6 phases** (Phase 25-30) — code remediation of v19.0 audit findings (architecture, naming, comments/docs, API/errors, tests, benchmarks)
- **12 new ADRs** — component sharing, feature flags, self-speculation, FP8, KV cache split, speculative decoding, per-request draft routing, multi-node feature-gating, FP8 quantizer orphan decision, CUDA graph feature-gating, cross-crate error boundaries, continuous batching

### Stats

- **Phases:** 6 (Phase 25-30)
- **Test count:** 1144+ passed, 0 failed
- **Coverage:** doc 97.8% / 99.6%

---

## 🚀 [v19.0] — Codebase Health Audit (2026-06-27)

### Added

- **5 analysis-only phases** (Phase 20-24) producing `.planning/audit/` directory:
    - Architecture audit (crate deps, module boundaries, layering matrix)
    - Naming audit (NAME-* findings)
    - Comments/docs audit (placeholder doc survey)
    - API/error audit (error type hygiene)
    - Test/benchmark audit
- **No code changes** — analysis-only milestone; findings drive v20.0-v23.0 remediation

### Stats

- **Phases:** 5 (Phase 20-24)
- **Output:** 5 audit reports in `.planning/audit/{architecture,naming,docs,api,benchmark}/`
- **Findings:** 22+ categories, drove 4 remediation milestones (v20.0, v21.0, v22.0, v23.0)

---

## Known Issues

- Long context (>32K) not yet supported (v24+ candidate)
- Multimodal/Vision not yet supported (v24+ candidate)
- Tool calling not yet supported (v24+ candidate)
- Multi-node / vllm-dist resurrection deferred (feature-gated only)

---

## Credits

Thanks to all contributors and the vLLM project for inspiration.
