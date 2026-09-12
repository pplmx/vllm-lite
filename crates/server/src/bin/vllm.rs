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

use anyhow::{Context, Result};
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
        /// Path to the YAML config file to validate against the server's
        /// own `AppConfig` rules.
        file: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    List {
        /// Directory to scan for model subdirectories.
        dir: PathBuf,
    },
    Info {
        /// Path to a model directory containing config.json.
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli {
        Cli::Config { command } => match command {
            ConfigCommand::Validate { file } => {
                validate_config(&file).context("validating config")?;
                println!("Config file is valid: {}", file.display());
            }
        },
        Cli::Model { command } => match command {
            ModelCommand::List { dir } => {
                list_models(&dir).context("listing models")?;
            }
            ModelCommand::Info { path } => {
                show_model_info(&path).context("showing model info")?;
            }
        },
    }
    Ok(())
}

fn validate_config(path: &PathBuf) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading config file {}", path.display()))?;
    // Parse with the SAME schema the server uses (`AppConfig` applies
    // `#[serde(default)]` to every section) and run the server's own
    // validator. Hand-rolling the checks here drifted from the server — the
    // tool rejected configs with no `server:`/`engine:` section (which the
    // server defaults) and missed the upper bounds (port, num_kv_blocks) the
    // server enforces — so `vllm config validate` must delegate or operators
    // get false rejections / false passes vs the running server.
    let config: vllm_server::config::AppConfig =
        serde_saphyr::from_str(&content).context("parsing config YAML syntax")?;
    if let Err(errors) = config.validate() {
        for err in &errors.0 {
            eprintln!("invalid: {err}");
        }
        anyhow::bail!("config validation failed with {} error(s)", errors.0.len());
    }

    Ok(())
}

fn list_models(dir: &PathBuf) -> Result<()> {
    if !dir.exists() {
        anyhow::bail!("directory does not exist: {}", dir.display());
    }

    println!("Available models in {}:", dir.display());
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Name                          │ Size      │ Type              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    for entry in std::fs::read_dir(dir).with_context(|| format!("reading dir {}", dir.display()))? {
        let entry = entry.context("reading directory entry")?;
        let path = entry.path();

        if path.is_dir()
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
        {
            let size = calculate_dir_size(&path);
            let size_str = format_size(size);
            let model_type = detect_model_type(&path);

            println!("║ {name:<29} │ {size_str:>9} │ {model_type:<17} ║");
        }
    }

    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}

fn show_model_info(path: &PathBuf) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("path does not exist: {}", path.display());
    }

    let config_path = path.join("config.json");
    let mut info = serde_json::Map::new();

    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        let config: Value = serde_json::from_str(&content)?;

        if let Some(obj) = config.as_object() {
            if let Some(model_type) = obj.get("model_type").or_else(|| obj.get("architectures")) {
                info.insert("architecture".to_string(), model_type.clone());
            }
            if let Some(hidden_size) = obj.get("hidden_size") {
                info.insert("hidden_size".to_string(), hidden_size.clone());
            }
            if let Some(num_layers) = obj.get("num_hidden_layers").or_else(|| obj.get("n_layers")) {
                info.insert("num_layers".to_string(), num_layers.clone());
            }
            if let Some(num_heads) = obj
                .get("num_attention_heads")
                .or_else(|| obj.get("n_heads"))
            {
                info.insert("num_heads".to_string(), num_heads.clone());
            }
            if let Some(vocab_size) = obj.get("vocab_size") {
                info.insert("vocab_size".to_string(), vocab_size.clone());
            }
            if let Some(eos) = obj.get("eos_token_id") {
                // RIL ISS-075: the model's end-of-sentence token — the server
                // now stops generation here (finish_reason=stop).
                info.insert("eos_token_id".to_string(), eos.clone());
            }
        }
    }

    let model_files = find_model_files(path);
    let total_size: u64 = model_files
        .iter()
        .map(|(p, _)| std::fs::metadata(p).map_or(0, |m| m.len()))
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
        println!("{key:>15}: {value}");
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

    // invariant: byte counts are bounded by available memory (<< 2^52 bytes),
    // so the u64 -> f64 conversion is lossless for display purposes.
    if bytes >= GB {
        #[allow(clippy::cast_precision_loss)]
        let value = bytes as f64 / GB as f64;
        format!("{value:.1} GB")
    } else if bytes >= MB {
        #[allow(clippy::cast_precision_loss)]
        let value = bytes as f64 / MB as f64;
        format!("{value:.1} MB")
    } else if bytes >= KB {
        #[allow(clippy::cast_precision_loss)]
        let value = bytes as f64 / KB as f64;
        format!("{value:.1} KB")
    } else {
        format!("{bytes} B")
    }
}

fn detect_model_type(path: &Path) -> &'static str {
    let config_path = path.join("config.json");
    let Ok(content) = std::fs::read_to_string(config_path) else {
        return "Unknown";
    };
    let Ok(config) = serde_json::from_str::<Value>(&content) else {
        return "Unknown";
    };
    let Some(name) = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(Value::as_str)
    else {
        return "Unknown";
    };

    if name.contains("Llama") {
        return "LLaMA";
    }
    if name.contains("Mistral") {
        return "Mistral";
    }
    if name.contains("Qwen") {
        return "Qwen";
    }
    if name.contains("Gemma") {
        return "Gemma";
    }
    if name.contains("Mixtral") {
        return "Mixtral";
    }

    "Unknown"
}

fn find_model_files(path: &PathBuf) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    let extensions = ["safetensors", "bin", "pt", "ckpt"];

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry_path.is_file()
                && let Some(ext) = entry_path.extension().and_then(|e| e.to_str())
                && extensions.contains(&ext)
            {
                let filename = entry_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                files.push((entry_path, filename));
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
    fn test_validate_config_missing_sections_use_defaults() {
        // A config with only a partial `server:` section (no `engine`) is
        // VALID — the server's AppConfig applies `#[serde(default)]` to every
        // section, so the validator must agree instead of issuing a false
        // rejection. Guards the RIL TASK-110 fix that replaced the hand-rolled
        // required-field check with the server's real `validate()`.
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

        assert!(validate_config(&config_path).is_ok());
    }

    #[test]
    fn test_validate_config_rejects_invalid_values() {
        // The server's validator (PortZero, KvBlocksTooLarge, ...) must be
        // surfaced by `vllm config validate` — a config the running server
        // would refuse must not pass the tool.
        let temp = TempDir::new().unwrap();
        let port_zero = temp.path().join("port_zero.yaml");
        std::fs::write(&port_zero, "server:\n  port: 0\n").unwrap();
        assert!(
            validate_config(&port_zero).is_err(),
            "port 0 must be rejected (server would refuse to start)"
        );

        let kv_too_large = temp.path().join("kv_too_large.yaml");
        std::fs::write(&kv_too_large, "engine:\n  num_kv_blocks: 70000\n").unwrap();
        assert!(
            validate_config(&kv_too_large).is_err(),
            "num_kv_blocks > 65536 must be rejected"
        );
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

    // RIL ISS-117: the documented usage (`vllm config validate config.yaml`)
    // was rejected because the args were `#[arg(long)]` flags. Guard the
    // positional form (and the `--<flag>` spelling docs never claimed) so the
    // tool's own usage comment stays true.
    #[test]
    fn config_validate_uses_positional_file() {
        let cli = Cli::try_parse_from(["vllm", "config", "validate", "config.yaml"])
            .expect("documented positional form must parse");
        match cli {
            Cli::Config {
                command: ConfigCommand::Validate { file },
            } => assert_eq!(file, PathBuf::from("config.yaml")),
            _ => panic!("expected config validate"),
        }
    }

    #[test]
    fn model_list_uses_positional_dir() {
        let cli = Cli::try_parse_from(["vllm", "model", "list", "/models"])
            .expect("documented positional form must parse");
        match cli {
            Cli::Model {
                command: ModelCommand::List { dir },
            } => assert_eq!(dir, PathBuf::from("/models")),
            _ => panic!("expected model list"),
        }
    }

    #[test]
    fn model_info_uses_positional_path() {
        let cli = Cli::try_parse_from(["vllm", "model", "info", "/models/llama-7b"])
            .expect("documented positional form must parse");
        match cli {
            Cli::Model {
                command: ModelCommand::Info { path },
            } => assert_eq!(path, PathBuf::from("/models/llama-7b")),
            _ => panic!("expected model info"),
        }
    }
}
