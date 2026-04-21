#[cfg(test)]
mod tests {
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
        use vllm_model::loader::format::{FormatLoader, SafetensorsLoader};
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
        use vllm_model::loader::format::load_checkpoint;
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
        println!("Embedding keys: {:?}", embed_keys);

        let lm_keys: Vec<_> = weights
            .keys()
            .filter(|k| k.contains("language_model") || k.contains("model."))
            .take(10)
            .collect();
        println!("Model keys sample: {:?}", lm_keys);
    }

    #[test]
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
        println!("\nRemapped Embedding keys: {:?}", embed_keys);

        let first_20: Vec<_> = remapped.keys().take(20).collect();
        println!("First 20 remapped keys: {:?}", first_20);

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
        use std::path::PathBuf;
        use vllm_model::tokenizer::Tokenizer;
        let tokenizer_path = PathBuf::from("/models/Qwen3-0.6B/tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("Failed to load tokenizer");
        let test_inputs = [
            "hi",
            "你好",
            "<|im_start|>user\nhi<|im_end|><|im_start|>assistant\n",
        ];
        for input in &test_inputs {
            let tokens = tokenizer.encode(input);
            let decoded = tokenizer.decode(&tokens);
            println!("\n=== Input: '{}' ===", input);
            println!("Tokens: {:?}", tokens);
            println!("Decoded: '{}'", decoded);
        }
    }

    #[test]
    #[ignore]
    fn test_qwen3_direct_inference() {
        let model_path = "/models/Qwen3-0.6B";
        let device = Device::Cpu;
        let loader = vllm_model::ModelLoader::builder(device.clone())
            .with_model_dir(model_path.to_string())
            .with_kv_blocks(256)
            .build()
            .expect("Failed to create loader");
        let mut model = loader.load_model().expect("Failed to load model");

        let tokens: Vec<u32> = vec![0u32, 151643, 9925];
        let positions: Vec<usize> = vec![0, 1, 2];
        let block_ids: Vec<usize> = vec![0, 0, 0];

        println!("\n=== Testing Prefill ===");
        println!("Input tokens: {:?}", tokens);

        let output = model
            .forward(&[1], &[tokens], &[positions], &[block_ids], &[0], &[true])
            .expect("Prefill forward failed");

        println!("Output: {:?}", output);
    }

    #[test]
    #[ignore]
    fn test_qwen3_weight_diagnostics() {
        let model_path = "/models/Qwen3-0.6B";
        let device = Device::Cpu;
        let weights = do_load_weights(model_path, &device).expect("Failed to load weights");

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
                println!("\n=== {} ===", key);
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
                println!("\n=== {} ===", key);
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
        let model_path = "/models/Qwen3-0.6B";
        let device = Device::Cpu;
        let weights = do_load_weights(model_path, &device).expect("Failed to load weights");

        for layer_idx in 0..3 {
            let q_norm_key = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
            let k_norm_key = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
            for key in &[&q_norm_key, &k_norm_key] {
                if let Some(w) = weights.get(key.as_str()) {
                    let mean = w.mean_all().expect("Failed to compute mean");
                    let abs_mean = w
                        .abs()
                        .expect("abs failed")
                        .mean_all()
                        .expect("Failed to compute abs mean");
                    println!("\n=== {} ===", key);
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
        let models = [
            ("/models/Qwen3-0.6B", "Qwen3"),
            ("/models/Qwen2.5-0.5B-Instruct", "Qwen2.5"),
            ("/models/DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1"),
        ];

        for (path, name) in &models {
            let device = Device::Cpu;
            match do_load_weights(path, &device) {
                Ok(weights) => println!("{}: Loaded {} weights", name, weights.len()),
                Err(e) => println!("{}: FAILED - {}", name, e),
            }
        }
    }
}
