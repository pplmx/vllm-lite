//! Unified checkpoint loading.
//!
//! This module provides a single entry point for loading model weights
//! from various formats using the FormatLoader trait.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Result, Tensor};

use super::format::SafetensorsLoader;
use super::io::{convert_tensor, find_safetensors_files, load_file_mmap_or_read};

pub fn load_checkpoint(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    if SafetensorsLoader::can_load(path) {
        return load_safetensors(path, device);
    }
    #[cfg(feature = "gguf")]
    {
        use super::format::{FormatLoader, GgufLoader};
        if GgufLoader::can_load(path) {
            return GgufLoader::load(path, device);
        }
    }

    Err(candle_core::Error::msg(format!(
        "No loader available for path: {}",
        path.display()
    )))
}

pub(crate) fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    use rayon::prelude::*;

    let files = if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        find_safetensors_files(path)?
    };

    let mmap_results: Vec<Result<(std::path::PathBuf, Vec<u8>)>> = files
        .par_iter()
        .map(|path| {
            let data = load_file_mmap_or_read(path)?;
            Ok((path.clone(), data))
        })
        .collect();

    let tensor_vec: Vec<Result<Vec<(String, Tensor)>>> = mmap_results
        .into_par_iter()
        .map(|result| {
            let (_path, data) = result?;
            let file = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| candle_core::Error::msg(format!("deserialize failed: {}", e)))?;
            file.tensors()
                .into_iter()
                .filter(|(name, _)| {
                    !name.contains("visual.") && !name.contains("vision_") && !name.contains("img_")
                })
                .map(|(name, view)| {
                    let tensor = convert_tensor(&view, device)?;
                    Ok((name, tensor))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect();

    let mut weights = HashMap::new();

    for result in tensor_vec {
        let tensors = result?;
        for (name, tensor) in tensors {
            if weights.insert(name.clone(), tensor).is_some() {
                return Err(candle_core::Error::msg(format!(
                    "Duplicate weight '{}'",
                    name
                )));
            }
        }
    }

    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_checkpoint_unsupported_format() {
        let temp_dir = TempDir::new().unwrap();
        let file = temp_dir.path().join("model.bin");
        std::fs::write(&file, b"test").unwrap();

        let result = load_checkpoint(&file, &Device::Cpu);
        assert!(result.is_err());
    }
}
