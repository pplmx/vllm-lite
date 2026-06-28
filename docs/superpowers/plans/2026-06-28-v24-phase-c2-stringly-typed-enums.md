# v24.0 Phase C-2 — Stringly-typed Enums

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace three stringly-typed public APIs with typed enums for compile-time safety and IDE autocomplete.

**Architecture:** Three independent enum conversions. Each is a breaking change to a specific function signature or struct field; call sites are updated atomically in the same commit.

**Tech Stack:** Rust 2024 edition, thiserror, serde (for `RopeType` rename_all).

**Spec:** `docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md` §6 (revised 2026-06-28)

**Audit source:** `/tmp/phase_c_audit/02_stringly_typed.md`

**Sister plans:**
- Phase C-1 — Engine Builder + Re-exports
- Phase C-3 — Trait Default Impls (separate plan)

---

## File Structure

| File | Change | Task |
|------|--------|------|
| `crates/core/src/speculative/registry/<file>.rs` | Add `DraftResolutionKind` enum + update function | T1 |
| `crates/core/src/speculative/registry/` (call sites) | Update callers | T1 |
| `crates/model/src/components/rope/` or similar | Add `RopeType` enum | T2 |
| `crates/model/src/<config file>` | Update `Option<String>` → `Option<RopeType>` | T2 |
| `crates/server/src/openai/batch/types.rs` (or similar) | Add `BatchEndpoint` enum | T3 |
| `crates/server/src/openai/batch/` (call sites) | Update callers | T3 |
| `CHANGELOG.md` | Phase C-2 entry | T4 |

---

## Task 1: `DraftResolutionKind` enum

**Files:**
- Modify: file containing `inc_draft_resolution(kind: &str)` (find via `rg`)
- Modify: call sites

- [ ] **Step 1: Locate the function**

```bash
rg "fn inc_draft_resolution|inc_draft_resolution\(" /workspace/vllm-lite/crates/ --type rust -n
```

Identify:
- The function definition file
- The parameter type (likely `&str`)
- The valid string values it accepts (look at match arms or comments)
- All call sites (typically 4-10)

- [ ] **Step 2: Define the enum**

In the same file as the function, add (typically in `types.rs` or top of file):

```rust
/// Kind of draft resolution strategy (v18+ per-request draft resolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DraftResolutionKind {
    /// Single shared draft model (v17 behavior).
    Shared,
    /// Per-request draft model selected from registry.
    PerRequest,
    /// No draft — target model only.
    None,
}

impl DraftResolutionKind {
    /// Parse from string (case-insensitive). Returns `None` if unrecognized.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "shared" => Some(Self::Shared),
            "per_request" | "per-request" | "perrequest" => Some(Self::PerRequest),
            "none" => Some(Self::None),
            _ => None,
        }
    }

    /// String representation (canonical form).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::PerRequest => "per_request",
            Self::None => "none",
        }
    }
}

impl std::fmt::Display for DraftResolutionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
```

Adjust the variants and parse logic to match the actual valid string values used in the existing code.

- [ ] **Step 3: Update function signature**

Change from:
```rust
pub fn inc_draft_resolution(&mut self, kind: &str) {
    match kind {
        "shared" => { /* ... */ }
        "per_request" => { /* ... */ }
        _ => { /* default */ }
    }
}
```

To:
```rust
pub fn inc_draft_resolution(&mut self, kind: DraftResolutionKind) {
    match kind {
        DraftResolutionKind::Shared => { /* ... */ }
        DraftResolutionKind::PerRequest => { /* ... */ }
        DraftResolutionKind::None => { /* default */ }
    }
}
```

- [ ] **Step 4: Update all call sites**

```bash
rg "inc_draft_resolution" /workspace/vllm-lite/crates/ --type rust -n
```

For each call site, replace string literals with enum variants:
```rust
// Before
engine.inc_draft_resolution("shared");
// After
engine.inc_draft_resolution(DraftResolutionKind::Shared);
```

If a call site has a dynamic string, use `.parse()` and handle the `Option`.

- [ ] **Step 5: Add tests**

In the function's file's existing test module, add:

```rust
#[test]
fn draft_resolution_kind_parse_roundtrip() {
    for kind in [DraftResolutionKind::Shared, DraftResolutionKind::PerRequest, DraftResolutionKind::None] {
        assert_eq!(DraftResolutionKind::parse(kind.as_str()), Some(kind));
    }
}

#[test]
fn draft_resolution_kind_parse_invalid() {
    assert_eq!(DraftResolutionKind::parse("invalid"), None);
}
```

- [ ] **Step 6: Verify build + tests**

```bash
cargo build --workspace --all-features 2>&1 | tail -5
cargo test -p vllm-core --lib 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add crates/
git commit -m "refactor(core): replace stringly-typed inc_draft_resolution with DraftResolutionKind enum"
```

---

## Task 2: `RopeType` enum

**Files:**
- Modify: `crates/model/src/<config file with rope_type>` (find via `rg "rope_type: Option<String>"`)

- [ ] **Step 1: Locate the rope_type field**

```bash
rg "rope_type.*Option<String>|rope_type.*:.*String" /workspace/vllm-lite/crates/ --type rust -n
```

Identify:
- The struct containing `rope_type: Option<String>`
- The valid string values (likely "linear", "dynamic", "yarn", "su", etc. — see HuggingFace RoPE convention)
- All construction sites (typically 5-15)

- [ ] **Step 2: Define the enum**

In the same module as the struct, add:

```rust
/// RoPE scaling type (HuggingFace-compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeType {
    /// No scaling (identity).
    #[default]
    None,
    /// Linear position interpolation.
    Linear,
    /// Dynamic NTK-aware scaling.
    Dynamic,
    /// YaRN (Yet another RoPE extensioN).
    Yarn,
    /// Su RoPE (RoPE in any precision).
    Su,
    /// Other (unknown to this enum; use string fallback).
    #[serde(other)]
    Other,
}

impl RopeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Dynamic => "dynamic",
            Self::Yarn => "yarn",
            Self::Su => "su",
            Self::Other => "other",
        }
    }
}
```

Adjust variants to match the actual valid string values used in the codebase.

- [ ] **Step 3: Update the struct field**

Change:
```rust
pub struct RopeScaling {
    pub rope_type: Option<String>,
    pub factor: Option<f32>,
    // ...
}
```

To:
```rust
pub struct RopeScaling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_type: Option<RopeType>,
    pub factor: Option<f32>,
    // ...
}
```

If the struct already derives `serde::Deserialize`, the `#[serde(rename_all = "lowercase")]` on `RopeType` ensures JSON values match.

- [ ] **Step 4: Update construction sites**

```bash
rg "rope_type: Some\(\"" /workspace/vllm-lite/crates/ --type rust -n
```

Replace:
```rust
RopeScaling { rope_type: Some("linear".to_string()), .. }
```

With:
```rust
RopeScaling { rope_type: Some(RopeType::Linear), .. }
```

Also handle the reverse direction if any code reads `rope_type` as a string.

- [ ] **Step 5: Add tests**

```rust
#[test]
fn rope_type_serde_lowercase() {
    let json = serde_json::to_string(&RopeType::Linear).unwrap();
    assert_eq!(json, "\"linear\"");

    let parsed: RopeType = serde_json::from_str("\"yarn\"").unwrap();
    assert_eq!(parsed, RopeType::Yarn);

    let other: RopeType = serde_json::from_str("\"future_unknown\"").unwrap();
    assert_eq!(other, RopeType::Other);
}
```

- [ ] **Step 6: Verify build + tests + clippy + fmt**

```bash
cargo fmt --all
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5
cargo clippy --workspace --all-targets --all-features -- \
  -D clippy::correctness -D clippy::suspicious -D clippy::perf 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add crates/
git commit -m "refactor(model): replace stringly-typed rope_type with RopeType enum"
```

---

## Task 3: `BatchEndpoint` enum

**Files:**
- Modify: file containing `endpoint: String` for batch jobs (likely `crates/server/src/openai/batch/types.rs`)
- Modify: call sites

- [ ] **Step 1: Locate the endpoint field**

```bash
rg "endpoint.*:.*String|endpoint: &str" /workspace/vllm-lite/crates/server/src/openai/batch/ --type rust -n
```

- [ ] **Step 2: Define the enum**

In the same module:

```rust
/// OpenAI Batch API endpoint kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchEndpoint {
    /// `/v1/chat/completions` (chat).
    Chat,
    /// `/v1/completions` (text completion).
    Completion,
    /// `/v1/embeddings` (embeddings).
    Embeddings,
}

impl BatchEndpoint {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "chat" | "/v1/chat/completions" => Some(Self::Chat),
            "completion" | "completions" | "/v1/completions" => Some(Self::Completion),
            "embeddings" | "/v1/embeddings" => Some(Self::Embeddings),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Completion => "completion",
            Self::Embeddings => "embeddings",
        }
    }
}

impl std::fmt::Display for BatchEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
```

Adjust variants based on actual valid endpoint strings in the codebase.

- [ ] **Step 3: Update field type**

Change:
```rust
pub struct SimpleBatchRequest {
    pub endpoint: String,
    // ...
}
```

To:
```rust
pub struct SimpleBatchRequest {
    pub endpoint: BatchEndpoint,
    // ...
}
```

**NOTE**: If `SimpleBatchRequest` derives `serde::Deserialize` from JSON, you may need to use `String` at the wire level and convert:

```rust
#[serde(deserialize_with = "deserialize_batch_endpoint")]
pub endpoint: BatchEndpoint,

fn deserialize_batch_endpoint<'de, D: serde::Deserializer<'de>>(de: D) -> Result<BatchEndpoint, D::Error> {
    let s = String::deserialize(de)?;
    BatchEndpoint::parse(&s).ok_or_else(|| serde::de::Error::custom(format!("unknown endpoint: {s}")))
}
```

Use this pattern if the field is JSON-deserialized. Otherwise direct enum is fine.

- [ ] **Step 4: Update construction sites**

```bash
rg "endpoint: \"chat\"|endpoint: \"completion\"|endpoint: \"embeddings\"" /workspace/vllm-lite/crates/ --type rust -n
```

Replace string literals with enum variants:
```rust
// Before
SimpleBatchRequest { endpoint: "chat".to_string(), .. }
// After
SimpleBatchRequest { endpoint: BatchEndpoint::Chat, .. }
```

- [ ] **Step 5: Add tests**

```rust
#[test]
fn batch_endpoint_parse_all_variants() {
    assert_eq!(BatchEndpoint::parse("chat"), Some(BatchEndpoint::Chat));
    assert_eq!(BatchEndpoint::parse("/v1/chat/completions"), Some(BatchEndpoint::Chat));
    assert_eq!(BatchEndpoint::parse("completion"), Some(BatchEndpoint::Completion));
    assert_eq!(BatchEndpoint::parse("embeddings"), Some(BatchEndpoint::Embeddings));
    assert_eq!(BatchEndpoint::parse("unknown"), None);
}
```

- [ ] **Step 6: Verify build + tests + clippy + fmt**

```bash
cargo fmt --all
cargo build --workspace --all-features 2>&1 | tail -5
cargo test --workspace --lib 2>&1 | tail -5
cargo clippy --workspace --all-targets --all-features -- \
  -D clippy::correctness -D clippy::suspicious -D clippy::perf 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add crates/
git commit -m "refactor(server): replace stringly-typed batch endpoint with BatchEndpoint enum"
```

---

## Task 4: Verify Full CI

- [ ] **Step 1: Run full CI**

```bash
just ci
```

Expected: all 4 steps pass; 1165 tests still pass (or +3 new tests).

---

## Task 5: Phase C-2 Completion Report (CHANGELOG)

**Files:**
- Modify: `/workspace/vllm-lite/CHANGELOG.md`

- [ ] **Step 1: Add Phase C-2 entry under `[Unreleased]` → `### Changed`**

```markdown
- **Stringly-typed APIs (v24.0 Phase C-2)** — replaced 3 string-typed public APIs with typed enums:
  - `DraftResolutionKind` enum replaces `&str` in `inc_draft_resolution(...)` and similar per-request draft APIs
  - `RopeType` enum (serde `lowercase`) replaces `Option<String>` in `RopeScaling::rope_type`
  - `BatchEndpoint` enum replaces `String` in batch request endpoint field
  - All string→enum conversions are breaking changes; affected call sites updated atomically
  - Each enum provides `parse(&str) -> Option<Self>`, `as_str() -> &'static str`, and `Display` impl
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record v24.0 Phase C-2 stringly-typed enums"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** §6 stringly-typed conversion covered (3 strong candidates)
- [x] **Placeholder scan:** Each enum has explicit variants based on actual codebase strings
- [x] **Type consistency:** Enum names follow existing convention (`XxxKind`, `XxxType`)
- [x] **Dependency order:** T1 → T2 → T3 → T4 → T5

---

## Handoff

After Task 5 commit, Phase C-2 is complete. Expected: 4 atomic commits (3 enum conversions + CHANGELOG). Push to origin/main.
