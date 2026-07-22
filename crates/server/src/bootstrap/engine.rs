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

/// Build the loader, model, optional draft model, and engine from CLI + config.
///
/// Returns the constructed engine and the model loader (the latter is retained
/// because the engine stores a reference to its architecture for routing).
#[allow(clippy::too_many_lines)]
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

    // Only load draft model if speculative decoding is enabled.
    let draft_model = if app_config.engine.max_draft_tokens > 0 {
        tracing::info!("Loading draft model (speculative decoding enabled)");
        Some(loader.load_model().context("failed to load draft model")?)
    } else {
        tracing::info!("Skipping draft model (speculative decoding disabled)");
        None
    };

    // Phase 41 OPS-32a second-half: when multi-node is enabled, route
    // through the `EngineBuilder` so the engine wires the
    // `PagedKvCacheWrapper` through `Engine::set_paged_kv_cache`.
    let engine = if app_config.server.multi_node.enabled {
        tracing::info!("Constructing Engine via EngineBuilder (multi-node path, Phase 41)");
        let mut builder = EngineBuilder::new(model);
        if let Some(d) = draft_model {
            builder = builder.with_draft_model(d);
        }
        builder = builder
            .with_config(SchedulerConfig::default())
            .with_num_kv_blocks(app_config.engine.num_kv_blocks)
            .with_max_draft_tokens(app_config.engine.max_draft_tokens);
        #[cfg(feature = "multi-node")]
        if let Some(cache) = loader.paged_kv_cache_clone() {
            builder = builder.with_paged_kv_cache(cache);
        }
        builder.build()
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
        Engine::new_boxed(model, draft_model)
    };

    Ok(engine)
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
