# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅                 |

## Reporting a Vulnerability

If you discover a security vulnerability in vLLM-lite, **open a private
GitHub Security Advisory** — that channel keeps the report confidential
until a fix ships and credits you in the release notes:

👉 <https://github.com/pplmx/vllm-lite/security/advisories/new>

For non-sensitive issues (typos in this policy, questions about
supported versions) open a regular GitHub issue instead.

For a list of current maintainers and their areas of responsibility see
[`MAINTAINERS.md`](./MAINTAINERS.md).

Please include the following information in the advisory:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (branch/commit)
- Any special configuration required to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Response Timeline

We aim to acknowledge vulnerability reports within **48 hours** and provide a more detailed response within **7 days**. We will work with the reporter to understand and resolve the issue, and keep them updated on our progress.

## Security Best Practices

- **API Keys**: Always use strong API keys in production
- **Rate Limiting**: Configure appropriate rate limits for your use case
- **Network**: Run behind a reverse proxy (nginx) for TLS termination
- **Updates**: Keep vLLM-lite updated to the latest version

## Scope

The following are **in scope** for security fixes:

- Authentication bypass
- Rate limiting bypass
- Memory safety issues
- Input validation vulnerabilities

The following are **out of scope** (handled externally):

- TLS/SSL → Configure at load balancer/reverse proxy
- DDoS protection → Use CDN or cloud DDoS protection
- Audit logging → Integrate with external SIEM

## Audit History

This section records the periodic `cargo audit` results and any remediation taken.

### 2026-06-28 (v26.0)

`cargo audit` baseline: 0 vulnerabilities + 2 warnings (rustls-pemfile, paste).
Dependabot alerts: 6 (1 high, 5 moderate).

| Crate            | Version  | Advisory                                              | Status                                                                                                                |
| ---------------- | -------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `rustls-pemfile` | 2.2.0    | RUSTSEC-2025-0134 (unmaintained, **high**)            | ✅ Migrated `tls.rs` to `rustls::pki_types::PemObject`; crate removed from deps                                       |
| `tower-http`     | 0.5/0.6  | Dependabot outdated (moderate)                        | ✅ Bumped to 0.7 (workspace-unified)                                                                                  |
| `serde_yaml`     | 0.9.34+deprecated | Dependabot deprecated (moderate)              | ✅ Migrated to pure-Rust `serde-saphyr 0.0.27` (panic-free, no libyaml C dep, no `unsafe`)                             |
| `tokio-rustls`   | 0.26.4   | Dependabot outdated (moderate)                        | ⚠️ Deferred — audit assumed 0.27 available, but registry only has 0.26.x. Will revisit when 0.27 ships upstream.       |
| `aws-lc-rs`      | 1.16.3   | Dependabot outdated (moderate)                        | ✅ Bumped to 1.17.0                                                                                                   |
| `tiktoken`       | 3.1.4    | (Dependabot; minor)                                   | ✅ Bumped to 3.5.1                                                                                                    |
| `hyper`          | 1.9.0    | (Dependabot; minor)                                   | ✅ Bumped to 1.10.1                                                                                                   |
| `paste`          | 1.0.15   | RUSTSEC-2024-0436 (unmaintained, INFO)                | ⚠️ **Accepted** — INFO severity (no vuln, no patch available); deep transitive via `gemm` → `candle-core`. Verified `candle-core 0.11.0` (latest, 2026-06-26) still depends on `gemm ^0.19` → `paste ^1.0`, so upgrade does not resolve. Suppressed in `just audit` via `--ignore RUSTSEC-2024-0436`. `cargo audit --strict` will still report it. |

Post-remediation audit: 1 INFO warning remaining (paste — accepted risk, see above).

CI workflow fix: removed broken `--all-features` from default GitHub Actions
`cargo clippy` job (no CUDA in default runners); switched to per-group denies
matching local `just clippy`. One follow-up const-correctness fix in test code
(`Qwen3Fixture::with_kv_blocks` now uses targeted `#[allow]` since the inner
type has a non-trivial Drop, making const-ineligibility structural).

Refs: `docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md`,
`/tmp/phase_f_audit/SUMMARY.md`

### 2026-06-28 (v27.0 — paste disposition)

**Audit source:** `/tmp/phase_g_audit/SUMMARY.md` (390 lines)

v26.0 plan deferred `paste` RUSTSEC-2024-0436 to "v27.0 candle-core upgrade" on the assumption that bumping candle would resolve the advisory. **Verified false**: `candle-core 0.11.0` (latest stable, released 2026-06-26) still depends on `gemm ^0.19.0` which still depends on `paste ^1.0`. No version of candle-core removes the `paste` transitive dependency.

**Disposition:** Accepted risk (informational-only advisory). `paste!` macro is tiny (~150 LoC), stable, used only by `gemm` for SIMD intrinsic codegen. RUSTSEC-2024-0436 is INFO severity (unmaintained, no patch, no exploitable vulnerability).

**Action:** suppress in `just audit` (local CI) via `--ignore RUSTSEC-2024-0436`; `just audit-strict` still reports it for awareness. No code change; no dependency change.

**Alternatives considered:**
- `pastey` (maintained fork): tested in audit — requires `[patch.crates-io]` override of `gemm`'s `paste` dep. Risk: gemm's FFI surface may break; cargo audit shows gemm internal type changes when `paste` semantics shift. Cost: 1-2 days. Not worth the risk for an INFO-severity advisory.
- candle-core upgrade: dead-end (still depends on paste).
- Drop `gemm` entirely: would require reimplementing BLAS integration for candle. Out of scope.

### 2026-06-26 (Wave 3)

`cargo audit` baseline: 0 vulnerabilities + 3 warnings.

| Crate            | Version | Advisory                                              | Status                                                                                                                |
| ---------------- | ------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `openssl`        | 0.10.79 | (Dependabot; no RUSTSEC)                              | ✅ Bumped to 0.10.80                                                                                                  |
| `memmap2`        | 0.9.10  | RUSTSEC-2026-0186 (unsound: unchecked pointer offset) | ✅ Bumped to 0.9.11                                                                                                   |
| `rustls-pemfile` | 2.2.0   | RUSTSEC-2025-0134 (unmaintained)                      | ⚠️ Deferred — 2.x line terminated; replacement requires `crates/server/src/security/tls.rs` API refactor              |
| `paste`          | 1.0.15  | RUSTSEC-2024-0436 (unmaintained)                      | ⚠️ Deferred — deep transitive via `gemm` → `candle-core`; resolution requires candle-core upgrade (independent scope) |

Post-remediation audit: 2 warnings remaining (rustls-pemfile + paste).

Refs: `docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md`
