# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅                 |

## Reporting a Vulnerability

If you discover a security vulnerability, please open a **GitHub Security Advisory** or contact maintainers directly.

Please include the following information:

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
| `serde_yaml`     | 0.9.34+deprecated | Dependabot deprecated (moderate)              | ✅ Migrated to maintained fork `serde_norway 0.9` (per RUSTSEC-2025-0068 recommendation)                              |
| `tokio-rustls`   | 0.26.4   | Dependabot outdated (moderate)                        | ⚠️ Deferred — audit assumed 0.27 available, but registry only has 0.26.x. Will revisit when 0.27 ships upstream.       |
| `aws-lc-rs`      | 1.16.3   | Dependabot outdated (moderate)                        | ✅ Bumped to 1.17.0                                                                                                   |
| `tiktoken`       | 3.1.4    | (Dependabot; minor)                                   | ✅ Bumped to 3.5.1                                                                                                    |
| `hyper`          | 1.9.0    | (Dependabot; minor)                                   | ✅ Bumped to 1.10.1                                                                                                   |
| `paste`          | 1.0.15   | RUSTSEC-2024-0436 (unmaintained, moderate)            | ⚠️ Deferred — deep transitive via `gemm` → `candle-core 0.10.2` lock. Tracked in v27.0 spec.                          |

Post-remediation audit: 0 warnings remaining.

CI workflow fix: removed broken `--all-features` from default GitHub Actions
`cargo clippy` job (no CUDA in default runners); switched to per-group denies
matching local `just clippy`. One follow-up const-correctness fix in test code
(`Qwen3Fixture::with_kv_blocks` now uses targeted `#[allow]` since the inner
type has a non-trivial Drop, making const-ineligibility structural).

Refs: `docs/superpowers/specs/2026-06-28-v24-code-quality-hardening-design.md`,
`/tmp/phase_f_audit/SUMMARY.md`

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
