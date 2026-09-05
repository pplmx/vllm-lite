// crates/server/src/bootstrap/engine.rs
//
// Engine construction: loader → model → optional draft model → Engine.
// Four construction paths are supported (chosen by config):
//   0. `EngineBuilder`       — multi-node path (Phase 41 OPS-32a second-half)
//   1. `with_budget_boxed`   — VRAM budget + draft specs (v18.0 path)
//   2. `with_drafts_boxed`   — draft specs only (v18.0 path, no budget)
//   3. `new_boxed`           — legacy path, no speculative config
//
// The `EngineBuilder` path is selected when
// `app_config.server.multi_node.enabled` is true so the engine wires
// the `PagedKvCache` through `EngineBuilder::with_paged_kv_cache`.
// All other paths are preserved for backward compatibility.
//
// Also includes `configure_speculative` which wires adaptive or vanilla
// speculative decoding onto a freshly constructed engine.

// `build_engine` / `configure_speculative` are the natural names for
// the bootstrap helpers in the `bootstrap::engine` module — the module
// name describes the concern, the function name describes the action.
#![allow(clippy::module_name_repetitions)]

use anyhow::{Context, Result};
use candle_core::Device;
use std::sync::Arc;
use vllm_core::engine::Engine;
use vllm_core::engine::EngineBuilder;
use vllm_core::types::{AdaptiveDraftConfig, SchedulerConfig};
use vllm_model::loader::ModelLoader;
use vllm_server::{cli, config::AppConfig};

/// Load the draft model when speculative decoding is enabled.
///
/// Returns `Some(model)` when `app_config.engine.max_draft_tokens > 0`,
/// or `None` (with a log line) when speculative decoding is disabled.
fn load_draft_model(
    loader: &ModelLoader,
    app_config: &AppConfig,
) -> Result<Option<Box<dyn vllm_traits::ModelBackend>>> {
    if app_config.engine.max_draft_tokens > 0 {
        tracing::info!("Loading draft model (speculative decoding enabled)");
        Ok(Some(
            loader.load_model().context("failed to load draft model")?,
        ))
    } else {
        tracing::info!("Skipping draft model (speculative decoding disabled)");
        Ok(None)
    }
}

/// Multi-node engine construction path (Phase 41).
///
/// Routes through `EngineBuilder` so the engine wires the
/// `PagedKvCacheWrapper` through `Engine::set_paged_kv_cache`.
/// Only active when `app_config.server.multi_node.enabled` is true.
fn build_engine_multi_node(
    model: Box<dyn vllm_traits::ModelBackend>,
    draft_model: Option<Box<dyn vllm_traits::ModelBackend>>,
    // `loader` is only referenced inside `#[cfg(feature = "multi-node")]`,
    // so it is unused under the default feature set. Suppress the lint
    // here rather than splitting the function or feature-gating the arg.
    #[allow(unused_variables)] loader: &ModelLoader,
    app_config: &AppConfig,
) -> Engine {
    tracing::info!("Constructing Engine via EngineBuilder (multi-node path, Phase 41)");
    let mut builder = EngineBuilder::new(model);
    if let Some(d) = draft_model {
        builder = builder.with_draft_model(d);
    }
    builder = builder
        .with_config(scheduler_config_from_app_config(app_config))
        .with_num_kv_blocks(app_config.engine.num_kv_blocks)
        .with_max_draft_tokens(app_config.engine.max_draft_tokens);
    #[cfg(feature = "multi-node")]
    if let Some(cache) = loader.paged_kv_cache_clone() {
        builder = builder.with_paged_kv_cache(cache);
    }
    builder.build()
}

/// Build the loader, model, optional draft model, and engine from CLI + config.
///
/// Returns the constructed engine and the model loader (the latter is retained
/// because the engine stores a reference to its architecture for routing).
pub fn build_engine(
    app_config: &AppConfig,
    cli: &cli::CliArgs,
) -> Result<(Engine, ModelLoader, Device)> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    tracing::info!(device = ?device, "Device initialized");

    let model_path = cli.model_path().display().to_string();
    tracing::debug!(model_path = %model_path, "Model path configured");

    let loader = ModelLoader::builder(device.clone())
        .with_model_dir(model_path.clone())
        .with_kv_blocks(app_config.engine.num_kv_blocks)
        .with_kv_quantization(app_config.engine.kv_quantization)
        .with_allow_stub(cli.model.allow_stub)
        .build()
        .context("failed to create model loader")?;

    let model = loader
        .load_model()
        .context("failed to load model weights")?;

    tracing::info!(
        model_path = %model_path,
        device = ?device,
        "Model loaded"
    );

    let draft_model = load_draft_model(&loader, app_config)?;

    // Phase 41 OPS-32a second-half: when multi-node is enabled, route
    // through the `EngineBuilder` so the engine wires the
    // `PagedKvCacheWrapper` through `Engine::set_paged_kv_cache`.
    let engine = if app_config.server.multi_node.enabled {
        build_engine_multi_node(model, draft_model, &loader, app_config)
    } else {
        build_engine_legacy(model, draft_model, app_config)?
    };

    Ok((engine, loader, device))
}

/// Legacy engine construction paths preserved for backward compatibility
/// (selected when `app_config.server.multi_node.enabled` is `false`).
fn build_engine_legacy(
    model: Box<dyn vllm_traits::ModelBackend>,
    draft_model: Option<Box<dyn vllm_traits::ModelBackend>>,
    app_config: &AppConfig,
) -> Result<Engine> {
    // v18.0: build the engine using with_budget_boxed / with_drafts_boxed when
    // the server config declares a VRAM budget or external draft specs. The
    // legacy new_boxed path is preserved for backward compatibility.
    let draft_specs: Vec<vllm_core::speculative::DraftSpec> = app_config
        .engine
        .draft_specs
        .iter()
        .map(|c| {
            let mut spec =
                vllm_core::speculative::DraftSpec::new(c.id.clone(), c.path.clone(), c.num_layers);
            if c.weight_size_bytes > 0 {
                spec = spec.with_weight_size(c.weight_size_bytes);
            }
            if let Some(arch) = &c.architecture {
                spec = spec.with_arch_hint(arch.clone());
            }
            spec
        })
        .collect();

    let engine = if let Some(budget_bytes) = app_config.engine.vram_budget_bytes {
        let budget = Arc::new(
            vllm_core::speculative::MemoryBudget::new(budget_bytes)
                .context("server config: invalid vram_budget_bytes")?,
        );
        tracing::info!(
            budget_bytes,
            draft_specs = draft_specs.len(),
            "Constructing Engine with VRAM budget (v18.0 path)"
        );
        Engine::with_budget_boxed(
            model,
            draft_model,
            draft_specs,
            budget,
            SchedulerConfig::default(),
            app_config.engine.max_draft_tokens,
            app_config.engine.num_kv_blocks,
        )
    } else if !draft_specs.is_empty() {
        tracing::info!(
            draft_specs = draft_specs.len(),
            "Constructing Engine with draft specs (v18.0 path, no budget)"
        );
        Engine::with_drafts_boxed(
            model,
            draft_model,
            draft_specs,
            SchedulerConfig::default(),
            app_config.engine.max_draft_tokens,
            app_config.engine.num_kv_blocks,
        )
    } else {
        // RIL ISS-081: the legacy fallback previously called
        // `Engine::new_boxed` which hardcodes max_draft_tokens=4 and
        // kv_blocks=1024, ignoring the configured values. Route it through
        // `with_config_boxed` so `--kv-blocks` / `--max-draft-tokens` /
        // `--max-batch-size` take effect here too. Default alignments:
        // kv_blocks default 1024 == new_boxed, so no change; max_draft
        // default 8 == the documented VLLM_MAX_DRAFT_TOKENS default (new_boxed
        // said 4 — an undocumented under-shoot now corrected to the docs).
        Engine::with_config_boxed(
            model,
            draft_model,
            scheduler_config_from_app_config(app_config),
            app_config.engine.max_draft_tokens,
            app_config.engine.num_kv_blocks,
        )
    };

    Ok(engine)
}

/// Build the `SchedulerConfig` used by every engine-construction path from
/// the server's `AppConfig` instead of `SchedulerConfig::default()`.
///
/// RIL ISS-081: `--max-batch-size` / `engine.max_batch_size` was parsed,
/// validated (1-8192) and stored but never consumed — every `build_engine*`
/// path passed `SchedulerConfig::default()`, so the knob changed nothing at
/// runtime. The engine's per-batch sequence cap is wired from
/// `SchedulerConfig::max_num_seqs` into the `BatchCompositionConfig`
/// (core/scheduler/engine/state/mod.rs), so the user-facing "max batch size"
/// knob maps there. Defaults coincide (`max_num_seqs` default 256 ==
/// `max_batch_size` default 256), so the mapping is a no-op unless the
/// operator raised or lowered the value.
fn scheduler_config_from_app_config(app_config: &AppConfig) -> SchedulerConfig {
    SchedulerConfig::builder()
        .with_max_num_seqs(app_config.engine.max_batch_size)
        .build()
}

/// Return startup warnings for engine knobs that are **documented but not
/// applied** by this build (RIL ISS-091) — an operator setting them today
/// gets a silent no-op, which is a config-honesty trap. Each returned
/// message is logged at `WARN` once at startup, before any traffic.
///
/// - `tensor_parallel_size > 1`: claims sharding that only the inert
///   `multi-node` / `vllm-dist` path could implement — this build runs a
///   single model worker on one device.
/// - `max_waiting_batches != default`: claims waiting-queue backpressure
///   the scheduler does not apply (it admits every queued request up to
///   the KV/memory budget).
///
/// Warnings fire only for **non-default** values: setting a knob to its
/// own default is a no-op by definition, and warning at every startup
/// would just be noise.
#[must_use]
pub fn inert_engine_knob_warnings(app_config: &AppConfig) -> Vec<String> {
    let mut warnings = Vec::new();
    if app_config.engine.tensor_parallel_size > 1 {
        warnings.push(format!(
            "engine.tensor_parallel_size = {} is NOT applied: this build runs a single \
             model worker on one device; tensor-parallel sharding (vllm-dist / multi-node) \
             is not wired, so the knob changes nothing (RIL ISS-091)",
            app_config.engine.tensor_parallel_size
        ));
    }
    if app_config.engine.max_waiting_batches != AppConfig::default().engine.max_waiting_batches {
        warnings.push(format!(
            "engine.max_waiting_batches = {} is NOT applied: the scheduler admits every \
             queued request up to the KV/memory budget and has no waiting-batch backpressure \
             in this build, so the knob changes nothing (RIL ISS-091)",
            app_config.engine.max_waiting_batches
        ));
    }
    warnings
}

/// Wire the checkpoint's end-of-sentence token into the engine (RIL ISS-075)
/// so a sequence stops as soon as the model emits it — `FinishReason::Stop`
/// instead of burning the remaining `max_tokens` budget.
///
/// Reads `eos_token_id` from the checkpoint `config.json` (the `HuggingFace`
/// convention; Qwen3 / Llama / Mistral checkpoints all carry it). Like
/// `configure_speculative`, this is a post-construction hook. The id may be a
/// single number or a list (some checkpoints declare several); the first
/// parseable entry wins. When the checkpoint declares none, EOS-stop stays
/// disabled and generation runs to `max_tokens` as before.
/// Read `eos_token_id` from a checkpoint `config.json` value, accepting the
/// `HuggingFace` conventions: a single integer (`"eos_token_id": 151645`) or a
/// list of integers (`[151645, ...]` — some checkpoints declare several). The
/// first parseable entry wins. `None` when absent or unparseable (stub/GGUF
/// checkpoints, or ids outside the `u32` token range).
#[must_use]
fn read_eos_token_id(config: &serde_json::Value) -> Option<u32> {
    config.get("eos_token_id").and_then(|v| match v {
        serde_json::Value::Number(n) => n.as_u64().and_then(|n| u32::try_from(n).ok()),
        serde_json::Value::Array(ids) => ids
            .iter()
            .find_map(|e| e.as_u64().and_then(|n| u32::try_from(n).ok())),
        _ => None,
    })
}

/// Wire the checkpoint's end-of-sentence token into the engine (RIL ISS-075)
/// so a sequence stops as soon as the model emits it — `FinishReason::Stop`
/// instead of burning the remaining `max_tokens` budget.
///
/// Like `configure_speculative`, this is a post-construction hook. When the
/// checkpoint declares no `eos_token_id`, EOS-stop stays disabled and
/// generation runs to `max_tokens` as before.
pub fn configure_eos(loader: &ModelLoader, engine: &mut Engine) {
    let eos_token_id = read_eos_token_id(loader.config_json());
    if let Some(eos) = eos_token_id {
        engine.set_eos_token_id(Some(eos));
        tracing::info!(eos_token_id = eos, "EOS-stop detection enabled");
    } else {
        tracing::debug!("checkpoint declares no eos_token_id; EOS-stop disabled");
    }
}

/// Wire optional speculative-decoding knobs onto a freshly constructed engine.
pub fn configure_speculative(app_config: &AppConfig, engine: &mut Engine) {
    if app_config.engine.max_draft_tokens > 0 {
        if app_config.engine.enable_adaptive_speculative {
            tracing::info!(
                "Enabling adaptive speculative decoding (max_draft_tokens={})",
                app_config.engine.max_draft_tokens
            );
            engine.enable_adaptive_speculative(AdaptiveDraftConfig {
                min_draft_tokens: 1,
                max_draft_tokens: app_config.engine.max_draft_tokens,
                target_acceptance_rate: 0.5,
                accuracy_window_size: 10,
                adjustment_step: 1,
                cooldown_steps: 5,
                ewma_alpha: 0.1,
                deadband_threshold: 0.05,
            });
        } else {
            tracing::info!(
                "Enabling speculative decoding (max_draft_tokens={})",
                app_config.engine.max_draft_tokens
            );
            engine.enable_speculative();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RIL ISS-091: the default configproduces no inert-knob warnings.
    #[test]
    fn inert_engine_knob_warnings_none_for_defaults() {
        let app = AppConfig::default();
        assert!(inert_engine_knob_warnings(&app).is_empty());
    }

    /// RIL ISS-091: `tensor_parallel_size > 1` is documented-but-inert and
    /// must warn at startup instead of silently doing nothing.
    #[test]
    fn inert_engine_knob_warnings_flags_tensor_parallel() {
        let mut app = AppConfig::default();
        app.engine.tensor_parallel_size = 4;
        let warnings = inert_engine_knob_warnings(&app);
        assert_eq!(warnings.len(), 1, "exactly the TP warning: {warnings:?}");
        assert!(warnings[0].contains("tensor_parallel_size"));
    }

    /// RIL ISS-091: a non-default `max_waiting_batches` is documented-but-
    /// inert and must warn at startup.
    #[test]
    fn inert_engine_knob_warnings_flags_max_waiting_batches() {
        let mut app = AppConfig::default();
        app.engine.max_waiting_batches = 50;
        let warnings = inert_engine_knob_warnings(&app);
        assert_eq!(
            warnings.len(),
            1,
            "exactly the waiting-batches warning: {warnings:?}"
        );
        assert!(warnings[0].contains("max_waiting_batches"));
    }

    /// RIL ISS-091: both inert knobs set → two warnings.
    #[test]
    fn inert_engine_knob_warnings_flags_both() {
        let mut app = AppConfig::default();
        app.engine.tensor_parallel_size = 2;
        app.engine.max_waiting_batches = 7;
        let warnings = inert_engine_knob_warnings(&app);
        assert_eq!(warnings.len(), 2, "{warnings:?}");
    }

    /// RIL ISS-075: the `eos_token_id` parser accepts the single-int
    /// `HuggingFace` form.
    #[test]
    fn read_eos_token_id_accepts_single_id() {
        let cfg: serde_json::Value = serde_json::json!({ "eos_token_id": 151_645 });
        assert_eq!(read_eos_token_id(&cfg), Some(151_645));
    }

    /// RIL ISS-075: some checkpoints declare `eos_token_id` as a list — the
    /// first parseable entry wins.
    #[test]
    fn read_eos_token_id_accepts_list_form() {
        let cfg: serde_json::Value = serde_json::json!({ "eos_token_id": [151_645, 151_643] });
        assert_eq!(read_eos_token_id(&cfg), Some(151_645));
    }

    /// RIL ISS-075: absent / non-numeric ids keep EOS-stop disabled (stub or
    /// GGUF checkpoints without `eos_token_id` — mirrors `max_model_len`'s
    /// `None`-when-undeclared contract).
    #[test]
    fn read_eos_token_id_returns_none_when_absent_or_invalid() {
        assert_eq!(read_eos_token_id(&serde_json::json!({})), None);
        assert_eq!(
            read_eos_token_id(&serde_json::json!({ "eos_token_id": "oops" })),
            None
        );
        assert_eq!(
            read_eos_token_id(&serde_json::json!({ "eos_token_id": ["oops"] })),
            None
        );
    }

    /// RIL ISS-081: `engine.max_batch_size` must reach the engine's
    /// scheduler (as `max_num_seqs`, the cap the batch composer is wired
    /// from). Pre-fix every construction path passed
    /// `SchedulerConfig::default()`, so the knob was inert.
    #[test]
    fn scheduler_config_wires_max_batch_size() {
        let mut app = AppConfig::default();
        app.engine.max_batch_size = 64;
        let cfg = scheduler_config_from_app_config(&app);
        assert_eq!(
            cfg.max_num_seqs, 64,
            "engine.max_batch_size must reach SchedulerConfig.max_num_seqs (got {})",
            cfg.max_num_seqs
        );
    }

    /// RIL ISS-081: lowering the knob must be honored; untouched scheduler
    /// knobs keep their documented defaults (not re-derived or zeroed).
    #[test]
    fn scheduler_config_preserves_defaults_and_honors_lowering() {
        let mut app = AppConfig::default();
        app.engine.max_batch_size = 4;
        let cfg = scheduler_config_from_app_config(&app);
        assert_eq!(cfg.max_num_seqs, 4);
        assert_eq!(cfg.max_num_batched_tokens, 4096);
        assert_eq!(cfg.max_consecutive_decode, 10);
        assert_eq!(cfg.prefill_chunk_size, 512);
        assert_eq!(cfg.max_batch_size, 256);
    }
}
