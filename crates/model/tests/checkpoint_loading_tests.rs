mod support;

#[cfg(test)]
mod tests {
    use super::support;
    use candle_core::Device;
    use std::path::Path;
    use vllm_model::loader::load_checkpoint;

    fn do_load_weights(
        model_dir: &str,
        device: &Device,
    ) -> candle_core::Result<std::collections::HashMap<String, candle_core::Tensor>> {
        load_checkpoint(Path::new(model_dir), device)
    }

    #[test]
    fn test_format_loader_trait_exists() {
        use vllm_model::loader::format::FormatLoader;
        let _ = std::any::type_name::<dyn FormatLoader>();
    }

    #[test]
    fn test_safetensors_loader_can_load() {
        use vllm_model::loader::SafetensorsLoader;
        let path = Path::new("model.safetensors");
        assert!(SafetensorsLoader::can_load(path));
        let path = Path::new("model.bin");
        assert!(!SafetensorsLoader::can_load(path));
    }

    #[test]
    fn test_storage_tensor_exists() {
        use vllm_model::quantize::StorageTensor;
        let _ = std::any::type_name::<StorageTensor>();
    }

    #[test]
    fn test_quantized_tensor_exists() {
        use vllm_model::quantize::QuantizedTensor;
        let _ = std::any::type_name::<QuantizedTensor>();
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn test_quantization_format_enum() {
        use vllm_model::quantize::QuantizationFormat;
        let format = QuantizationFormat::GgufQ4_K_M;
        assert_eq!(format, QuantizationFormat::GgufQ4_K_M);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn test_gguf_loader_can_load() {
        use vllm_model::loader::format::{FormatLoader, GgufLoader};
        let path = Path::new("model.gguf");
        assert!(GgufLoader::can_load(path));
        let path = Path::new("model.safetensors");
        assert!(!GgufLoader::can_load(path));
    }

    #[test]
    fn test_model_loader_uses_new_checkpoint_loading() {
        use candle_core::Device;
        use std::path::Path;
        use vllm_model::loader::load_checkpoint;
        let result = load_checkpoint(Path::new("/nonexistent"), &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_qwen35_weight_keys() {
        let model_path = "/models/Qwen3.5-0.8B";
        let device = Device::Cpu;
        let weights = do_load_weights(model_path, &device).expect("Failed to load weights");

        let embed_keys: Vec<_> = weights
            .keys()
            .filter(|k| k.contains("embed") || k.contains("token") || k.contains("embedding"))
            .collect();
        println!("Embedding keys: {embed_keys:?}");

        let lm_keys: Vec<_> = weights
            .keys()
            .filter(|k| k.contains("language_model") || k.contains("model."))
            .take(10)
            .collect();
        println!("Model keys sample: {lm_keys:?}");
    }

    #[test]
    #[ignore = "slow: on-disk Qwen3.5 weight inspection (run: just nextest-checkpoint)"]
    fn test_qwen35_full_weight_structure() {
        let model_path = "/models/Qwen3.5-0.8B";
        let device = Device::Cpu;
        let weights = do_load_weights(model_path, &device).expect("Failed to load weights");

        let layer0_keys: Vec<_> = weights.keys().filter(|k| k.contains("layers.0")).collect();
        println!("Layer 0 (linear_attention) keys:");
        for k in &layer0_keys {
            if let Some(w) = weights.get(*k) {
                println!("  {}: {:?}", k, w.dims());
            }
        }

        let layer3_keys: Vec<_> = weights.keys().filter(|k| k.contains("layers.3")).collect();
        println!("\nLayer 3 (full_attention) keys:");
        for k in &layer3_keys {
            if let Some(w) = weights.get(*k) {
                println!("  {}: {:?}", k, w.dims());
            }
        }

        let final_keys: Vec<_> = weights
            .keys()
            .filter(|k| k.contains("norm") || k.contains("lm_head") || k.contains("final"))
            .collect();
        println!("\nFinal layer keys:");
        for k in &final_keys {
            if let Some(w) = weights.get(*k) {
                println!("  {}: {:?}", k, w.dims());
            }
        }
    }

    #[test]
    #[ignore]
    fn test_qwen35_remapped_weight_structure() {
        use vllm_model::qwen3_5::arch::remap_qwen35_weight_keys;

        let model_path = "/models/Qwen3.5-0.8B";
        let device = Device::Cpu;
        let weights = do_load_weights(model_path, &device).expect("Failed to load weights");

        let remapped = remap_qwen35_weight_keys(weights);

        let embed_keys: Vec<_> = remapped.keys().filter(|k| k.contains("embed")).collect();
        println!("\nRemapped Embedding keys: {embed_keys:?}");

        let first_20: Vec<_> = remapped.keys().take(20).collect();
        println!("First 20 remapped keys: {first_20:?}");

        let layer0_keys: Vec<_> = remapped
            .keys()
            .filter(|k| {
                k.contains("layers.") && (k.contains("0.linear") || k.contains("layers.00"))
            })
            .collect();
        println!("\nRemapped Layer 0 linear_attn keys:");
        for k in &layer0_keys {
            if let Some(w) = remapped.get(*k) {
                println!("  {}: {:?}", k, w.dims());
            }
        }

        let full_attn_keys: Vec<_> = remapped
            .keys()
            .filter(|k| k.contains("self_attn"))
            .take(10)
            .collect();
        println!("\nRemapped self_attn keys (sample):");
        for k in &full_attn_keys {
            if let Some(w) = remapped.get(*k) {
                println!("  {}: {:?}", k, w.dims());
            }
        }
    }

    #[test]
    #[ignore]
    fn test_qwen3_tokenizer_roundtrip() {
        let tokenizer = support::qwen3::tokenizer();
        let test_inputs = [
            "hi",
            "你好",
            "<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n",
        ];
        for input in &test_inputs {
            let tokens = tokenizer.encode(input);
            let decoded = tokenizer.decode(&tokens);
            println!("\n=== Input: '{input}' ===");
            println!("Tokens: {tokens:?}");
            println!("Decoded: '{decoded}'");
        }
    }

    #[test]
    #[ignore]
    fn test_qwen3_direct_inference() {
        let mut model = support::qwen3::Qwen3Fixture::cpu()
            .with_kv_blocks(256)
            .load_model()
            .expect("Failed to load model");

        let tokens: Vec<u32> = vec![0u32, 151643, 9925];
        let positions: Vec<usize> = vec![0, 1, 2];
        let block_ids: Vec<usize> = vec![0, 0, 0];

        println!("\n=== Testing Prefill ===");
        println!("Input tokens: {tokens:?}");

        let output = model
            .forward(&[1], &[tokens], &[positions], &[block_ids], &[0], &[true])
            .expect("Prefill forward failed");

        println!("Output: {output:?}");
    }

    #[test]
    #[ignore]
    fn test_qwen3_weight_diagnostics() {
        let weights = support::qwen3::Qwen3Fixture::cpu()
            .checkpoint()
            .expect("Failed to load weights");

        let key_weights = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ];
        for key in &key_weights {
            if let Some(w) = weights.get(*key) {
                let mean = w.mean_all().expect("Failed to compute mean");
                let abs_mean = w
                    .abs()
                    .expect("abs failed")
                    .mean_all()
                    .expect("Failed to compute abs mean");
                println!("\n=== {key} ===");
                println!("Shape: {:?}", w.dims());
                println!(
                    "Mean: {:.6}, Abs Mean: {:.6}",
                    mean.to_scalar::<f32>().unwrap_or(0.0),
                    abs_mean.to_scalar::<f32>().unwrap_or(0.0)
                );
            }
        }

        let layer_weights = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ];
        for key in &layer_weights {
            if let Some(w) = weights.get(*key) {
                let mean = w.mean_all().expect("Failed to compute mean");
                let abs_mean = w
                    .abs()
                    .expect("abs failed")
                    .mean_all()
                    .expect("Failed to compute abs mean");
                println!("\n=== {key} ===");
                println!("Shape: {:?}", w.dims());
                println!(
                    "Mean: {:.6}, Abs Mean: {:.6}",
                    mean.to_scalar::<f32>().unwrap_or(0.0),
                    abs_mean.to_scalar::<f32>().unwrap_or(0.0)
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn test_qwen3_qk_norm_weights() {
        let weights = support::qwen3::Qwen3Fixture::cpu()
            .checkpoint()
            .expect("Failed to load weights");

        for layer_idx in 0..3 {
            let q_norm_key = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
            let k_norm_key = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
            for key in &[&q_norm_key, &k_norm_key] {
                if let Some(w) = weights.get(key.as_str()) {
                    let mean = w.mean_all().expect("Failed to compute mean");
                    let abs_mean = w
                        .abs()
                        .expect("abs failed")
                        .mean_all()
                        .expect("Failed to compute abs mean");
                    println!("\n=== {key} ===");
                    println!("Shape: {:?}", w.dims());
                    println!(
                        "Mean: {:.6}, Abs Mean: {:.6}",
                        mean.to_scalar::<f32>().unwrap_or(0.0),
                        abs_mean.to_scalar::<f32>().unwrap_or(0.0)
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_all_models_loadable() {
        let qwen3_dir = support::qwen3::model_dir();
        let models = [
            (qwen3_dir.as_path(), "Qwen3"),
            (Path::new("/models/Qwen2.5-0.5B-Instruct"), "Qwen2.5"),
            (
                Path::new("/models/DeepSeek-R1-0528-Qwen3-8B"),
                "DeepSeek-R1",
            ),
        ];

        for (path, name) in &models {
            let device = Device::Cpu;
            match load_checkpoint(path, &device) {
                Ok(weights) => println!("{}: Loaded {} weights", name, weights.len()),
                Err(e) => println!("{name}: FAILED - {e}"),
            }
        }
    }
}
