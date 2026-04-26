//! CLI Tool for vLLM-lite
//!
//! Provides utility commands for managing models and validating configurations.
//!
//! ## Usage
//!
//! ```bash
//! # Validate a config file
//! cargo run --bin vllm -- config validate config.yaml
//!
//! # List available models
//! cargo run --bin vllm -- model list /models
//!
//! # Show model metadata
//! cargo run --bin vllm -- model info /models/llama-7b
//! ```

use clap::{Parser, Subcommand};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "vllm")]
#[command(version = "0.1.0")]
#[command(about = "vLLM-lite CLI tools", long_about = None)]
enum Cli {
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCommand {
    Validate {
        #[arg(long)]
        file: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    List {
        #[arg(long)]
        dir: PathBuf,
    },
    Info {
        #[arg(long)]
        path: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli {
        Cli::Config { command } => match command {
            ConfigCommand::Validate { file } => {
                if let Err(e) = validate_config(&file) {
                    eprintln!("Config validation failed: {}", e);
                    std::process::exit(1);
                }
                println!("Config file is valid: {}", file.display());
            }
        },
        Cli::Model { command } => match command {
            ModelCommand::List { dir } => {
                if let Err(e) = list_models(&dir) {
                    eprintln!("Failed to list models: {}", e);
                    std::process::exit(1);
                }
            }
            ModelCommand::Info { path } => {
                if let Err(e) = show_model_info(&path) {
                    eprintln!("Failed to get model info: {}", e);
                    std::process::exit(1);
                }
            }
        },
    }
}

fn validate_config(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let parsed: Value =
        serde_yaml::from_str(&content).map_err(|e| format!("YAML syntax error: {}", e))?;

    let required_fields = ["server", "engine"];
    for field in required_fields {
        if parsed.get(field).is_none() {
            return Err(format!("Missing required field: {}", field).into());
        }
    }

    if let Some(server) = parsed.get("server").and_then(|v| v.as_object()) {
        if let Some(port) = server.get("port") {
            if let Some(p) = port.as_i64() {
                if !(1..=65535).contains(&p) {
                    return Err(format!("Invalid port: {}", p).into());
                }
            }
        }
    }

    if let Some(engine) = parsed.get("engine").and_then(|v| v.as_object()) {
        if let Some(kv_blocks) = engine.get("num_kv_blocks") {
            if let Some(n) = kv_blocks.as_i64() {
                if n < 1 {
                    return Err(format!("Invalid kv_blocks: {}", n).into());
                }
            }
        }
    }

    Ok(())
}

fn list_models(dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.exists() {
        return Err(format!("Directory does not exist: {}", dir.display()).into());
    }

    println!("Available models in {}:", dir.display());
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Name                          │ Size      │ Type              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let size = calculate_dir_size(&path);
                let size_str = format_size(size);
                let model_type = detect_model_type(&path);

                println!("║ {:<29} │ {:>9} │ {:<17} ║", name, size_str, model_type);
            }
        }
    }

    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}

fn show_model_info(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("Path does not exist: {}", path.display()).into());
    }

    let config_path = path.join("config.json");
    let mut info = serde_json::Map::new();

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        let config: Value = serde_json::from_str(&content)?;

        if let Some(obj) = config.as_object() {
            if let Some(model_type) = obj.get("model_type").or(obj.get("architectures")) {
                info.insert("architecture".to_string(), model_type.clone());
            }
            if let Some(hidden_size) = obj.get("hidden_size") {
                info.insert("hidden_size".to_string(), hidden_size.clone());
            }
            if let Some(num_layers) = obj.get("num_hidden_layers").or(obj.get("n_layers")) {
                info.insert("num_layers".to_string(), num_layers.clone());
            }
            if let Some(num_heads) = obj.get("num_attention_heads").or(obj.get("n_heads")) {
                info.insert("num_heads".to_string(), num_heads.clone());
            }
            if let Some(vocab_size) = obj.get("vocab_size") {
                info.insert("vocab_size".to_string(), vocab_size.clone());
            }
        }
    }

    let model_files = find_model_files(path);
    let total_size: u64 = model_files
        .iter()
        .map(|(p, _)| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    info.insert(
        "total_size".to_string(),
        serde_json::json!(format_size(total_size)),
    );
    info.insert(
        "num_files".to_string(),
        serde_json::json!(model_files.len()),
    );

    println!(
        "Model: {}",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    println!("═══════════════════════════════════════════════════");
    for (key, value) in &info {
        println!("{:>15}: {}", key, value);
    }

    Ok(())
}

fn calculate_dir_size(path: &PathBuf) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += calculate_dir_size(&entry.path());
                }
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn detect_model_type(path: &Path) -> &'static str {
    let config_path = path.join("config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<Value>(&content) {
                if let Some(architectures) = config.get("architectures") {
                    if let Some(arr) = architectures.as_array() {
                        if let Some(first) = arr.first() {
                            if let Some(name) = first.as_str() {
                                if name.contains("Llama") {
                                    return "LLaMA";
                                } else if name.contains("Mistral") {
                                    return "Mistral";
                                } else if name.contains("Qwen") {
                                    return "Qwen";
                                } else if name.contains("Gemma") {
                                    return "Gemma";
                                } else if name.contains("Mixtral") {
                                    return "Mixtral";
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    "Unknown"
}

fn find_model_files(path: &PathBuf) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    let extensions = ["safetensors", "bin", "pt", "ckpt"];

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Some(ext) = entry_path.extension().and_then(|e| e.to_str()) {
                    if extensions.contains(&ext) {
                        let filename = entry_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        files.push((entry_path, filename));
                    }
                }
            }
        }
    }
    files.sort_by(|a, b| a.1.cmp(&b.1));
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validate_config_valid() {
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.yaml");
        std::fs::write(
            &config_path,
            r#"
server:
  host: "0.0.0.0"
  port: 8000
engine:
  num_kv_blocks: 1024
"#,
        )
        .unwrap();

        assert!(validate_config(&config_path).is_ok());
    }

    #[test]
    fn test_validate_config_missing_field() {
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.yaml");
        std::fs::write(
            &config_path,
            r#"
server:
  host: "0.0.0.0"
"#,
        )
        .unwrap();

        assert!(validate_config(&config_path).is_err());
    }

    #[test]
    fn test_validate_config_invalid_yaml() {
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.yaml");
        std::fs::write(&config_path, "invalid: yaml: content:").unwrap();

        assert!(validate_config(&config_path).is_err());
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    }
}
