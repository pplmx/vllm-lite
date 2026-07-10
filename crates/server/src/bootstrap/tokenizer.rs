// crates/server/src/bootstrap/tokenizer.rs
//
// Tokenizer loading from `<model_dir>/tokenizer.json`, with graceful
// fallback to a default-constructed tokenizer if the file is missing,
// not valid UTF-8, or fails to parse. Returns the `Arc<Tokenizer>` ready
// for use in `ApiState`.

use std::path::Path;
use std::sync::Arc;
use vllm_model::tokenizer::Tokenizer;

/// Load the tokenizer from `<model_dir>/tokenizer.json`, or fall back to a
/// default-constructed tokenizer. Returns the `Arc<Tokenizer>` ready for use.
pub fn load_tokenizer(model_dir: &Path) -> Arc<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        tracing::warn!("No tokenizer.json found in model directory, using default tokenizer");
        return Arc::new(Tokenizer::new());
    }
    let Some(path_str) = tokenizer_path.to_str() else {
        tracing::error!(
            path = ?tokenizer_path,
            "Tokenizer path is not valid UTF-8; falling back to default tokenizer"
        );
        return Arc::new(Tokenizer::new());
    };
    match Tokenizer::from_file(path_str) {
        Ok(t) => {
            tracing::info!("Tokenizer loaded");
            Arc::new(t)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to load tokenizer from file, using default");
            Arc::new(Tokenizer::new())
        }
    }
}
