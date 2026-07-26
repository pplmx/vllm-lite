// crates/server/src/bootstrap/tokenizer.rs
//
// Tokenizer loading from `<model_dir>/tokenizer.json`, with graceful
// fallback to a default-constructed tokenizer if the file is missing,
// not valid UTF-8, or fails to parse. Returns the `Arc<Tokenizer>` ready
// for use in `ApiState`.

// `load_tokenizer` is the natural name for the bootstrap helper in
// `bootstrap::tokenizer` — the module name describes the concern, the
// function name describes the action.
#![allow(clippy::module_name_repetitions)]

use std::path::Path;
use std::sync::Arc;
use vllm_model::tokenizer::Tokenizer;

/// Construct a default (fallback) tokenizer with logging.
fn default_tokenizer() -> Arc<Tokenizer> {
    tracing::warn!("Using default tokenizer (fallback)");
    Arc::new(Tokenizer::new())
}

/// Resolve the tokenizer path from `model_dir`, returning `None` with a
/// warning if the file is missing or not valid UTF-8.
fn resolve_tokenizer_path(model_dir: &Path) -> Option<String> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        tracing::warn!("No tokenizer.json found in model directory, using default tokenizer");
        return None;
    }
    let Some(path_str) = tokenizer_path.to_str() else {
        tracing::error!(
            path = ?tokenizer_path,
            "Tokenizer path is not valid UTF-8; falling back to default tokenizer"
        );
        return None;
    };
    Some(path_str.to_string())
}

/// Load the tokenizer from `<model_dir>/tokenizer.json`, or fall back to a
/// default-constructed tokenizer. Returns the `Arc<Tokenizer>` ready for use.
pub fn load_tokenizer(model_dir: &Path) -> Arc<Tokenizer> {
    let Some(path_str) = resolve_tokenizer_path(model_dir) else {
        return default_tokenizer();
    };
    match Tokenizer::from_file(&path_str) {
        Ok(t) => {
            tracing::info!("Tokenizer loaded");
            Arc::new(t)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to load tokenizer from file, using default");
            default_tokenizer()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_tokenizer_missing_file_returns_default() {
        // A directory with no tokenizer.json should fall back to the default.
        let dir = TempDir::new().unwrap();
        let tokenizer = load_tokenizer(dir.path());
        // Default tokenizer has model_name = None (no HF backend loaded).
        assert_eq!(tokenizer.model_name(), None);
    }

    #[test]
    fn test_load_tokenizer_invalid_json_returns_default() {
        // A tokenizer.json with invalid content should fall back to default.
        let dir = TempDir::new().unwrap();
        let tokenizer_path = dir.path().join("tokenizer.json");
        std::fs::write(&tokenizer_path, "not valid json").unwrap();
        let tokenizer = load_tokenizer(dir.path());
        assert_eq!(tokenizer.model_name(), None);
    }

    #[test]
    fn test_load_tokenizer_empty_directory_returns_default() {
        // An empty directory (no tokenizer.json) should return the default.
        let dir = TempDir::new().unwrap();
        let tokenizer = load_tokenizer(dir.path());
        // The default tokenizer encodes by splitting on whitespace.
        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_load_tokenizer_nonexistent_dir_returns_default() {
        // A non-existent directory should also fall back to default
        // (tokenizer.json won't exist there).
        let dir = TempDir::new().unwrap();
        let nonexistent = dir.path().join("nonexistent_subdir");
        let tokenizer = load_tokenizer(&nonexistent);
        assert_eq!(tokenizer.model_name(), None);
    }
}
