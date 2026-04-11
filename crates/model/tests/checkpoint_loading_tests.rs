#[cfg(test)]
mod tests {
    use std::path::Path;
    use vllm_model::loader::FormatLoader;

    #[test]
    fn test_format_loader_trait_exists() {
        use vllm_model::loader::format::FormatLoader;
        // Just checking trait exists
    }

    #[test]
    fn test_safetensors_loader_can_load() {
        use vllm_model::loader::format::SafetensorsLoader;

        let path = Path::new("model.safetensors");
        assert!(SafetensorsLoader::can_load(path));

        let path = Path::new("model.bin");
        assert!(!SafetensorsLoader::can_load(path));
    }
}
