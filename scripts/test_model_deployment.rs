//! Test script for model deployment verification
//!
//! Usage: cargo run --example test_model_deployment -- <model_path>

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("/models/Qwen3-0.6B");

    println!("Testing model deployment...");
    println!("Model path: {}", model_path);

    // Check if model directory exists
    if !Path::new(model_path).exists() {
        eprintln!("Error: Model directory does not exist: {}", model_path);
        std::process::exit(1);
    }

    // Check for required files
    let required_files = ["config.json", "model.safetensors"];
    for file in &required_files {
        let path = Path::new(model_path).join(file);
        if path.exists() {
            let size = std::fs::metadata(&path)?.len();
            println!("✓ Found {} ({:.2} MB)", file, size as f64 / 1_048_576.0);
        } else {
            eprintln!("✗ Missing required file: {}", file);
            std::process::exit(1);
        }
    }

    // Try to load config
    let config_path = Path::new(model_path).join("config.json");
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_content)?;

    println!("\nModel Configuration:");
    println!(
        "  Architecture: {}",
        config["architectures"][0].as_str().unwrap_or("unknown")
    );
    println!(
        "  Model Type: {}",
        config["model_type"].as_str().unwrap_or("unknown")
    );
    println!(
        "  Hidden Size: {}",
        config["hidden_size"].as_u64().unwrap_or(0)
    );
    println!(
        "  Num Layers: {}",
        config["num_hidden_layers"].as_u64().unwrap_or(0)
    );
    println!(
        "  Vocab Size: {}",
        config["vocab_size"].as_u64().unwrap_or(0)
    );

    // Calculate model size
    let model_file = Path::new(model_path).join("model.safetensors");
    let model_size = std::fs::metadata(&model_file)?.len();
    println!(
        "\nModel Size: {:.2} GB",
        model_size as f64 / 1_073_741_824.0
    );

    // Check if sharded
    let index_path = Path::new(model_path).join("model.safetensors.index.json");
    if index_path.exists() {
        println!("Model is sharded (multiple files)");
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;
        if let Some(weight_map) = index["weight_map"].as_object() {
            println!(
                "  Total shards: {} unique files",
                weight_map
                    .values()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            );
        }
    } else {
        println!("Model is single file");
    }

    // Estimate memory requirements
    let dtype_bytes = match config["torch_dtype"].as_str() {
        Some("float32") => 4,
        Some("float16") | Some("bfloat16") => 2,
        _ => 2, // assume 2 bytes for unknown
    };
    let estimated_memory_gb = (model_size as f64 * dtype_bytes as f64) / 1_073_741_824.0;
    println!("\nEstimated Memory Requirements:");
    println!("  Model weights: {:.2} GB", estimated_memory_gb);
    println!(
        "  KV Cache: ~{:.2} GB (varies by config)",
        estimated_memory_gb * 0.3
    );
    println!("  Total: ~{:.2} GB", estimated_memory_gb * 1.3);

    // Check available system memory
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let kb: u64 = parts[1].parse().unwrap_or(0);
                        let gb = kb as f64 / 1_048_576.0;
                        println!("\nSystem Memory: {:.2} GB", gb);
                        if gb < estimated_memory_gb * 1.5 {
                            println!("⚠ Warning: System memory may be insufficient");
                        } else {
                            println!("✓ System memory is sufficient");
                        }
                    }
                    break;
                }
            }
        }
    }

    println!("\n✓ Model deployment verification passed!");
    println!("  Model is ready for inference");

    Ok(())
}
