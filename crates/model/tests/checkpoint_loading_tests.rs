#[cfg(test)]
mod tests {
    use std::path::Path;

    #[test]
    fn test_format_loader_trait_exists() {
        use vllm_model::loader::format::FormatLoader;
        // Just checking trait exists
        let _ = std::any::type_name::<dyn FormatLoader>();
    }

    #[test]
    fn test_safetensors_loader_can_load() {
        use vllm_model::loader::format::{FormatLoader, SafetensorsLoader};

        let path = Path::new("model.safetensors");
        assert!(SafetensorsLoader::can_load(path));

        let path = Path::new("model.bin");
        assert!(!SafetensorsLoader::can_load(path));
    }

    #[test]
    fn test_storage_tensor_exists() {
        use vllm_model::quantize::StorageTensor;
        // Just checking type exists
        let _ = std::any::type_name::<StorageTensor>();
    }

    #[test]
    fn test_quantized_tensor_exists() {
        use vllm_model::quantize::QuantizedTensor;
        // Just checking type exists
        let _ = std::any::type_name::<QuantizedTensor>();
    }

    #[test]
    fn test_quantization_format_enum() {
        use vllm_model::quantize::QuantizationFormat;
        let format = QuantizationFormat::GgufQ4_K_M;
        assert_eq!(format, QuantizationFormat::GgufQ4_K_M);
    }

    #[test]
    fn test_gguf_loader_can_load() {
        use vllm_model::loader::format::{FormatLoader, GgufLoader};
        let path = Path::new("model.gguf");
        assert!(GgufLoader::can_load(path));

        let path = Path::new("model.safetensors");
        assert!(!GgufLoader::can_load(path));
    }
}
