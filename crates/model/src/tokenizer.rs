//! Tokenizer wrapper around `tokenizers` (`HuggingFace`) + chat-template rendering.
//!
//! `Tokenizer` is constructed from the `tokenizer.json` + `tokenizer_config.json`
//! shipped alongside the model weights. Chat templates are loaded from
//! `chat_template.jinja` if present; otherwise we fall back to the
//! built-in Qwen / Llama / Mistral templates.
#![allow(clippy::module_name_repetitions)]
use std::fmt::Write;
use tokenizers::Tokenizer as HFTokenizer;

/// Error type for tokenizer encoding/decoding failures. Covers invalid UTF-8, missing vocab entries, and chat-template substitution errors.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer from {path}: {source}")]
    LoadFailed {
        path: String,
        #[source]
        source: tokenizers::Error,
    },
}

#[derive(Debug)]
/// Tokenizer wrapper around `HuggingFace` `tokenizers::Tokenizer`.
///
/// Provides helpers for prompt + chat-template encoding. Constructed via
/// `Tokenizer::from_file` or `Tokenizer::from_pretrained`; never
/// instantiated directly because the inner HF tokenizer must be
/// deserialized from a JSON file.
pub struct Tokenizer {
    inner: Option<Box<HFTokenizer>>,
    vocab_size: usize,
    special_tokens: Vec<String>,
    model_name: Option<String>,
}

impl Tokenizer {
    /// Create a fallback tokenizer with the Qwen3 default vocabulary (no HF backend).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: None,
            vocab_size: 151_936,
            special_tokens: vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
            ],
            model_name: None,
        }
    }

    /// Construct a tokenizer from an already-built HF tokenizer.
    ///
    /// Mirrors [`Self::from_file`]'s field population (vocab size, added
    /// `<...>` special tokens) but takes the HF tokenizer in hand — the
    /// seam tests need when exercising a specific tokenizer shape (e.g.
    /// the byte-level BPE that splits a multi-byte char across tokens,
    /// RIL ISS-105/ISS-121) without a tokenizer.json on disk.
    #[must_use]
    pub fn from_hf_tokenizer(tokenizer: HFTokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true);
        let mut special_tokens = Vec::new();
        for id in tokenizer.get_added_tokens_decoder().keys() {
            if let Some(token) = tokenizer.id_to_token(*id)
                && !token.starts_with('▁')
                && token.len() > 1
                && token.starts_with('<')
            {
                special_tokens.push(token);
            }
        }
        if special_tokens.is_empty() {
            special_tokens = vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
            ];
        }
        Self {
            inner: Some(Box::new(tokenizer)),
            vocab_size,
            special_tokens,
            model_name: None,
        }
    }

    /// Construct a tokenizer from a tokenizer.json file.
    /// # Errors
    ///
    /// Returns `Err` if reading or parsing the source fails.
    pub fn from_file(path: &str) -> std::result::Result<Self, TokenizerError> {
        let tokenizer = HFTokenizer::from_file(path).map_err(|e| TokenizerError::LoadFailed {
            path: path.to_string(),
            source: e,
        })?;
        let vocab_size = tokenizer.get_vocab_size(true);
        // RIL ISS-147: derive the model id from the `--model` directory
        // (the parent of `tokenizer.json`), never a hardcoded constant —
        // a fabricated identity would make `/v1/models` and
        // `/health/details` advertise the wrong model to operations.
        let model_name = derive_model_name_from_path(path);

        let mut special_tokens = Vec::new();
        for id in tokenizer.get_added_tokens_decoder().keys() {
            if let Some(token) = tokenizer.id_to_token(*id)
                && !token.starts_with('▁')
                && token.len() > 1
                && token.starts_with('<')
            {
                special_tokens.push(token);
            }
        }

        if special_tokens.is_empty() {
            special_tokens = vec![
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
            ];
        }

        Ok(Self {
            inner: Some(Box::new(tokenizer)),
            vocab_size,
            special_tokens,
            model_name,
        })
    }

    /// Encode `text` into token IDs using the HF tokenizer, or an ASCII fallback if none is loaded.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(ref tokenizer) = self.inner
            && let Ok(encoding) = tokenizer.encode(text, false)
        {
            return encoding.get_ids().to_vec();
        }

        text.split_whitespace()
            .enumerate()
            .map(|(i, _)| u32::try_from(i + 1).unwrap_or(u32::MAX))
            .collect()
    }

    /// Decode token IDs back to text via the HF tokenizer, or an ASCII placeholder if none is loaded.
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        if let Some(ref tokenizer) = self.inner
            && let Ok(text) = tokenizer.decode(tokens, false)
        {
            return text;
        }

        tokens.iter().fold(String::new(), |mut acc, t| {
            let _ = write!(acc, "token_{t} ");
            acc
        })
    }

    /// Return the vocabulary size (number of token IDs).
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Return the list of special tokens (e.g. `<|im_start|>`, `<|im_end|>`).
    #[must_use]
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    /// Returns `true` if `text` matches one of the registered special tokens.
    #[must_use]
    pub fn is_special_token(&self, text: &str) -> bool {
        self.special_tokens.iter().any(|t| t.as_str() == text)
    }

    /// Strip all special tokens from `text` and return the trimmed remainder.
    #[must_use]
    pub fn clean_special_tokens(&self, text: &str) -> String {
        let mut result = text.to_string();
        for token in &self.special_tokens {
            result = result.replace(token.as_str(), "");
        }
        result.trim().to_string()
    }

    /// Return the model name parsed from the tokenizer config, if available.
    #[must_use]
    pub fn model_name(&self) -> Option<String> {
        self.model_name.clone()
    }
}

/// Derive the model id from a `tokenizer.json` path (RIL ISS-147).
///
/// `Tokenizer::from_file` is given `<model_dir>/tokenizer.json`, so the
/// parent directory's basename is exactly the `--model` directory the
/// operator pointed at — the honest identity to report through
/// `/v1/models` and `/health/details`. Returns `None` for a bare
/// filename with no parent (unusual; defensive).
fn derive_model_name_from_path(path: &str) -> Option<String> {
    std::path::Path::new(path)
        .parent()
        .and_then(|dir| dir.file_name())
        .map(|name| name.to_string_lossy().into_owned())
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental token-stream decoder that never splits a multi-byte UTF-8
/// character across SSE chunks (RIL ISS-105).
///
/// The naive streaming path decodes each sampled token in isolation
/// (`tokenizer.decode(&[t])`). For byte-level BPE tokenizers (the shipped
/// Qwen defaults) a single Unicode char can be split across two
/// consecutive byte-merges; HF's `ByteLevel` decoder flushes its pending
/// prefix as `U+FFFD` at the end of each `decode` call, so the stream
/// emits corruption (`�`) while the non-streaming path (full-list decode)
/// is correct. This is exactly the mismatch observed for split chars.
///
/// Strategy (tokenizer-agnostic — no byte-level knowledge needed): keep
/// the un-emitted tokens pending; decode the window, and only emit a
/// prefix whose decode does **not** end in a lossy `U+FFFD`. A partial
/// char at the window tail always lossy-decodes to a trailing `U+FFFD`,
/// so that tail is held back and re-decoded once the next token arrives.
/// A *genuine* `U+FFFD` emitted by the model (complete `EF BF BD` bytes)
/// is also held for one step and then re-decodes identically — delayed by
/// one token, never corrupted. Deterministic for every tokenizer format.
#[derive(Debug, Default)]
pub struct StreamingDecoder {
    /// Tokens received since the last emission that have not yet been
    /// confirmed to end on a complete character boundary.
    pending: Vec<u32>,
}

impl StreamingDecoder {
    /// Create an empty streaming decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Handle the next sampled token, returning the decodable output
    /// emitted at this step (possibly empty while awaiting the bytes that
    /// complete a multi-byte char).
    #[must_use]
    pub fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> String {
        self.pending.push(token);
        // Find the longest prefix that decodes with no trailing lossy
        // replacement char. Typical step: 0–1 iterations (the last token
        // is almost always complete); a mid-char split needs 1–2 more.
        let mut emit_inclusive = self.pending.len(); // exclusive end
        while emit_inclusive > 0 {
            let s = tokenizer.decode(&self.pending[..emit_inclusive]);
            if !s.ends_with('\u{FFFD}') {
                break;
            }
            emit_inclusive -= 1;
        }
        let emitted = tokenizer.decode(&self.pending[..emit_inclusive]);
        self.pending.drain(..emit_inclusive);
        emitted
    }

    /// Flush any held-back suffix (stream end). Emits the remaining tokens
    /// as-is — at the true end of the stream there is no future byte to
    /// complete a partial char, so whatever the lossy decode yields is the
    /// best available output.
    #[must_use]
    pub fn flush(&mut self, tokenizer: &Tokenizer) -> String {
        let out = tokenizer.decode(&self.pending);
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = Tokenizer::new();
        let _ = tokenizer.encode("test");
    }

    #[test]
    fn test_tokenizer_encode_simple() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("hello world");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[1], 2);
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 151_936);
    }

    // RIL ISS-147: the model id reported by `Tokenizer::from_file` must
    // be derived from the model directory (the `--model` dir the operator
    // pointed at), never a hardcoded constant — the pre-fix tokenizer
    // reported `"Qwen3.5-0.8B"` for EVERY real checkpoint (a Llama or
    // Mistral deploy advertised a fabricated Qwen id to `/v1/models`,
    // `/v1/models/{id}`, and `/health/details`). The derivation helper is
    // tested directly (no tokenizer.json needed on disk).
    #[test]
    fn test_derive_model_name_from_path() {
        // A tokenizer inside a named model dir → the dir's basename.
        assert_eq!(
            crate::tokenizer::derive_model_name_from_path(
                "/models/Qwen2.5-0.5B-Instruct/tokenizer.json"
            ),
            Some("Qwen2.5-0.5B-Instruct".to_string())
        );
        // A relative model dir works too.
        assert_eq!(
            crate::tokenizer::derive_model_name_from_path(
                "checkpoints/llama-3.2-1b/tokenizer.json"
            ),
            Some("llama-3.2-1b".to_string())
        );
        // A bare filename (no directory) has no derivable id.
        assert_eq!(
            crate::tokenizer::derive_model_name_from_path("tokenizer.json"),
            None
        );
    }

    #[test]
    fn test_tokenizer_encode_empty() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("");

        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_single_word() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.encode("hello");

        assert_eq!(tokens.len(), 1);
    }

    #[test]
    fn test_tokenizer_decode_single_token() {
        let tokenizer = Tokenizer::new();
        let text = tokenizer.decode(&[1]);

        assert!(!text.is_empty());
    }

    #[test]
    fn test_tokenizer_qwen3_hi_token() {
        use std::path::PathBuf;

        let model_path = PathBuf::from("/models/Qwen3-0.6B");
        let tokenizer_path = model_path.join("tokenizer.json");

        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                // invariant: pre-conditions make this infallible at this call site.
                .expect("Failed to load tokenizer");

            let tokens = tokenizer.encode("hi");
            assert_eq!(tokens.len(), 1, "hi should be a single token");
            assert_eq!(tokens[0], 6023, "hi should be token 6023");

            let decoded = tokenizer.decode(&tokens);
            assert!(decoded.contains("hi"), "Decoded text should contain 'hi'");
        }
    }

    #[test]
    fn test_tokenizer_qwen3_chat_prompt() {
        use std::path::PathBuf;

        let model_path = PathBuf::from("/models/Qwen3-0.6B");
        let tokenizer_path = model_path.join("tokenizer.json");

        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                // invariant: pre-conditions make this infallible at this call site.
                .expect("Failed to load tokenizer");

            let im_start = "<|im_start|>";
            let im_end = "<|im_end|>";
            let prompt = format!(
                "{}user\nhi{}{}\n{}assistant\n",
                im_start, im_end, "\n", im_start
            );

            let tokens = tokenizer.encode(&prompt);
            assert!(!tokens.is_empty(), "Chat prompt should produce tokens");
            assert!(
                tokens.len() < 100,
                "Chat prompt should be reasonable length"
            );

            let decoded = tokenizer.decode(&tokens);
            assert!(!decoded.is_empty(), "Should be able to decode tokens");
        }
    }
}

#[cfg(test)]
mod streaming_decode_tests {
    use super::*;
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPT;
    use tokenizers::tokenizer::Tokenizer as HFT;

    /// Reproduce the GPT2/byte-level byte→char encoding (`bytes_char()`
    /// in the tokenizers crate): printable ASCII → itself, Latin-1
    /// 0xA1..=0xAC and 0xAE..=0xFF → itself, every other byte b → the
    /// char 0x100+n in encounter order. Used to build vocab entries the
    /// `ByteLevel` decoder maps back to raw bytes.
    fn byte_char(b: u8) -> char {
        if (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b) {
            char::from(b)
        } else {
            // Count prior non-self bytes to compute the 0x100+n slot.
            let n = (0..b)
                .filter(|x| {
                    !(0x21..=0x7E).contains(x)
                        && !(0xA1..=0xAC).contains(x)
                        && !(0xAE..=0xFF).contains(x)
                })
                .count() as u32;
            char::from_u32(0x100 + n).unwrap_or('\u{FFFD}')
        }
    }

    /// Build a byte-level BPE whose vocab contains the raw bytes of U+4F60
    /// (你, bytes E4 BD A0) as three SEPARATE tokens (the merges never
    /// combined them) — the scenario where a streamed generation emits a
    /// multi-byte char across consecutive tokens. Wrapped in the
    /// `vllm_model::tokenizer::Tokenizer` so the production decode path is
    /// exercised end-to-end.
    fn split_char_tokenizer() -> Tokenizer {
        let mut vocab = ahash::AHashMap::default();
        vocab.insert("h".to_string(), 0);
        vocab.insert("i".to_string(), 1);
        vocab.insert(byte_char(0xe4).to_string(), 2);
        vocab.insert(byte_char(0xbd).to_string(), 3);
        vocab.insert(byte_char(0xa0).to_string(), 4);
        let mut hf = HFT::new(
            BPE::builder()
                .vocab_and_merges(vocab, tokenizers::models::bpe::Merges::default())
                .build()
                .unwrap(),
        );
        hf.with_pre_tokenizer(Some(ByteLevelPT::default()));
        hf.with_decoder(Some(ByteLevelDecoder::default()));
        Tokenizer::from_hf_tokenizer(hf)
    }

    /// Regression for RIL ISS-105: streaming the three split-byte tokens
    /// one at a time through `StreamingDecoder` must reassemble 你 — the
    /// naive per-token `decode(&[t])` path produced `���`.
    #[test]
    fn streaming_decoder_reassembles_split_char() {
        let tokenizer = split_char_tokenizer();
        // Prove the premise: whole-list decode is correct...
        assert_eq!(tokenizer.decode(&[2, 3, 4]), "你");
        // ...but naive per-token decode corrupts it.
        assert_eq!(
            format!(
                "{}{}{}",
                tokenizer.decode(&[2]),
                tokenizer.decode(&[3]),
                tokenizer.decode(&[4])
            ),
            "���",
            "premise: per-token decode must split the char (what streaming fix replaces)"
        );

        let mut dec = StreamingDecoder::new();
        let mut out = String::new();
        for tok in [2u32, 3, 4] {
            out.push_str(&dec.push(&tokenizer, tok));
        }
        out.push_str(&dec.flush(&tokenizer));
        assert_eq!(
            out, "你",
            "streaming decoder must reassemble the split char: got {out:?}"
        );
    }

    /// Streaming every token must equal the whole-list decode for a normal
    /// (already-merged) multi-byte sequence, and buffered output must not
    /// introduce stray characters.
    #[test]
    fn streaming_decoder_matches_whole_list_decode() {
        let tokenizer = split_char_tokenizer();
        let seq = [0u32, 1, 2, 3, 4, 0, 1]; // "hi" + 你 + "hi"
        let expected = tokenizer.decode(&seq);
        let mut dec = StreamingDecoder::new();
        let mut out = String::new();
        for &tok in &seq {
            out.push_str(&dec.push(&tokenizer, tok));
        }
        out.push_str(&dec.flush(&tokenizer));
        assert_eq!(out, expected, "streaming must equal whole-list decode");
    }

    /// A genuine U+FFFD emitted by the model (complete EF BF BD bytes) is
    /// held for at most one step but never corrupted, and never blocks the
    /// following complete token. In a byte-level vocab each raw byte has
    /// exactly one token id, so the BD byte of 你 and the BD byte of the
    /// literal "�" are the SAME token (id 3).
    #[test]
    fn streaming_decoder_preserves_genuine_replacement_char() {
        // tokens: h(0) i(1) | 你 = [2,3,4] | literal "�" = [EF,BF,BD] where
        // EF=6 and BF=7 are not yet in the base vocab — add them, reusing
        // BD=3 for the shared third byte.
        let mut v = ahash::AHashMap::default();
        v.insert("h".to_string(), 0);
        v.insert("i".to_string(), 1);
        v.insert(byte_char(0xe4).to_string(), 2);
        v.insert(byte_char(0xbd).to_string(), 3);
        v.insert(byte_char(0xa0).to_string(), 4);
        v.insert(byte_char(0xef).to_string(), 6);
        v.insert(byte_char(0xbf).to_string(), 7);
        let mut hf = HFT::new(
            BPE::builder()
                .vocab_and_merges(v, tokenizers::models::bpe::Merges::default())
                .build()
                .unwrap(),
        );
        hf.with_pre_tokenizer(Some(ByteLevelPT::default()));
        hf.with_decoder(Some(ByteLevelDecoder::default()));
        let tokenizer = Tokenizer {
            inner: Some(Box::new(hf)),
            vocab_size: 8,
            special_tokens: Vec::new(),
            model_name: None,
        };
        // 你 (tokens 2,3,4) followed by a genuine replacement char
        // (tokens 6,7,3 = EF BF BD) — the literal "你�".
        let stream = [2u32, 3, 4, 6, 7, 3];
        let expected = tokenizer.decode(&stream);
        assert_eq!(expected, "你\u{FFFD}");
        let mut dec = StreamingDecoder::new();
        let mut out = String::new();
        for tok in stream {
            out.push_str(&dec.push(&tokenizer, tok));
        }
        out.push_str(&dec.flush(&tokenizer));
        assert_eq!(out, "你\u{FFFD}", "genuine U+FFFD must survive streaming");
    }
}
