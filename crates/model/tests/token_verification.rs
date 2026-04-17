use vllm_model::tokenizer::Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    fn is_printable_text(s: &str) -> bool {
        !s.is_empty() && !s.chars().any(|c| c == '\u{FFFD}') && s.chars().any(|c| c.is_alphabetic())
    }

    fn setup_tokenizer() -> Tokenizer {
        let path = std::path::PathBuf::from("/models/Qwen3-0.6B/tokenizer.json");
        if !path.exists() {
            panic!("Tokenizer not found at {:?}", path);
        }
        Tokenizer::from_file(path.to_str().unwrap()).expect("Failed to load tokenizer")
    }
}
