//! Model loading utilities.
//!
//! This module provides the ModelLoader for loading model weights and configurations.
//! Supports Builder pattern for flexible configuration.

pub mod builder;
pub mod checkpoint;
pub mod format;
pub mod io;

pub use builder::{ModelLoader, ModelLoaderBuilder};
pub use checkpoint::load_checkpoint;
pub use format::{FormatLoader, SafetensorsLoader};

#[cfg(test)]
mod tests {
    use super::{load_checkpoint, SafetensorsLoader};
    use candle_core::Device;
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn test_load_checkpoint_nonexistent_path() {
        let path = Path::new("/nonexistent/path/model");
        let result = load_checkpoint(path, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_loader_can_load_by_extension() {
        let path = Path::new("model.safetensors");
        assert!(SafetensorsLoader::can_load(path));
    }

    #[test]
    fn test_safetensors_loader_cannot_load_other_extensions() {
        assert!(!SafetensorsLoader::can_load(Path::new("model.bin")));
        assert!(!SafetensorsLoader::can_load(Path::new("model.pt")));
        assert!(!SafetensorsLoader::can_load(Path::new("model.ckpt")));
    }

    #[test]
    fn test_safetensors_loader_can_load_directory_with_file() {
        let temp_dir = TempDir::new().unwrap();
        std::fs::write(temp_dir.path().join("model.safetensors"), b"test").unwrap();
        assert!(SafetensorsLoader::can_load(temp_dir.path()));
    }

    #[test]
    fn test_safetensors_loader_cannot_load_empty_directory() {
        let temp_dir = TempDir::new().unwrap();
        assert!(!SafetensorsLoader::can_load(temp_dir.path()));
    }
}
