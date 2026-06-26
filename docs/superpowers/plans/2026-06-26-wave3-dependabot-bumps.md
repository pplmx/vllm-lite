# Wave 3: Dependabot 漏洞修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** bump `openssl 0.10.79 → 0.10.80` 和 `memmap2 0.9.10 → 0.9.11`；记录 deferred 项；同步 SECURITY.md 与 SESSION-HANDOFF。

**Architecture:** 仅 Cargo.lock 变更（无代码改动）；文档同步按 Wave 1/2 同样模式。

**Tech Stack:** cargo, cargo-audit

**基线 commit:** `c93ba5e`（spec 已落地）

**前置验证:**

```bash
cd /workspace/vllm-lite
just nextest         # 必须 ≥ 1035 passed
cargo clippy --workspace --all-targets -- -D warnings  # 必须绿
git log --oneline -1 # 应为 c93ba5e

# 确认 cargo-audit 已装
which cargo-audit 2>&1
# 预期: /root/.cargo/bin/cargo-audit（已装）
```

---

## Task 1: openssl + memmap2 bump（1 commit）

**Files:**
- Modify: `Cargo.lock`

- [ ] **Step 1: 记录 baseline audit 状态**

```bash
cd /workspace/vllm-lite
cargo audit 2>&1 | grep -E "^Crate:|^Warning:|^Title:" | head -20 > /tmp/wave3-baseline-audit.txt
cat /tmp/wave3-baseline-audit.txt
# 预期: 列出 3 个 warnings (memmap2, rustls-pemfile, paste)
```

- [ ] **Step 2: bump openssl 到 0.10.80**

```bash
cd /workspace/vllm-lite
cargo update -p openssl --precise 0.10.80
# 预期: "Updating crates.io index" + 更新 openssl 0.10.79 → 0.10.80（openssl-sys 0.9.115 → 0.9.116 同步）
```

- [ ] **Step 3: bump memmap2 到 0.9.11**

```bash
cd /workspace/vllm-lite
cargo update -p memmap2
# 预期: 自动选 0.9.11（latest）
```

- [ ] **Step 4: 验证 build（含 tests）**

```bash
cd /workspace/vllm-lite
cargo build --workspace
cargo build --workspace --tests
# 预期: 全部成功。若 memmap2 0.9.11 与 candle-core ABI 不兼容，build 失败 → 见风险缓解
```

If build fails (rare but possible memmap2 ABI break):
```bash
# Revert memmap2 bump only
cargo update -p memmap2 --precise 0.9.10
# Continue with only openssl bump
```

- [ ] **Step 5: 验证 clippy + nextest**

```bash
cd /workspace/vllm-lite
cargo clippy --workspace --all-targets -- -D warnings
just nextest
# 预期: clippy 0 errors; nextest ≥ 1035 passed（无回归）
```

- [ ] **Step 6: 重跑 audit 确认 warning 减少**

```bash
cd /workspace/vllm-lite
cargo audit 2>&1 | grep -E "^Crate:|^Warning:|^Title:" > /tmp/wave3-post-audit.txt
cat /tmp/wave3-post-audit.txt
# 预期: 应仅剩 rustls-pemfile + paste（2 个 warnings；memmap2 + openssl 已清）
diff /tmp/wave3-baseline-audit.txt /tmp/wave3-post-audit.txt
# 预期: memmap2 行被删除
```

- [ ] **Step 7: 验证 Cargo.lock diff 范围**

```bash
cd /workspace/vllm-lite
git diff --stat Cargo.lock
# 预期: 1 file changed, 6-10 insertions, 6-10 deletions
#       （openssl + openssl-sys + memmap2 各 2 行变化）

git diff Cargo.lock | grep -E "^[+-]" | grep -E "openssl|memmap2"
# 预期: 仅 openssl 与 memmap2 相关行有变化；无其他无关变更
```

- [ ] **Step 8: Commit**

```bash
cd /workspace/vllm-lite
git add Cargo.lock
git commit -m "chore(deps): bump openssl 0.10.79 -> 0.10.80 and memmap2 0.9.10 -> 0.9.11

- openssl 0.10.80: patch bump; closes Dependabot alert (transitive via native-tls -> hyper-tls -> metrics-exporter-prometheus / reqwest)
- openssl-sys 0.9.115 -> 0.9.116: dependency sync
- memmap2 0.9.11: patch bump; likely fixes RUSTSEC-2026-0186 (unchecked pointer offset); transitive via candle-core

Deferred (document-only):
- rustls-pemfile 2.2.0: RUSTSEC-2025-0134 unmaintained; 2.x line terminated; replacement requires tls.rs API refactor (out of Wave 3 scope)
- paste 1.0.15: RUSTSEC-2024-0436 unmaintained; deep transitive via gemm -> candle-core; requires candle-core upgrade (independent spec)

Refs: docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md"
```

---

## Task 2: SECURITY.md 加 Audit History 段（1 commit）

**Files:**
- Modify: `SECURITY.md`

- [ ] **Step 1: 读当前 SECURITY.md 末尾结构**

```bash
cd /workspace/vllm-lite
tail -20 SECURITY.md
# 预期: 现有 "Scope" 段已结束；文件末尾可能有空行
```

- [ ] **Step 2: 在文件末尾追加 Audit History 段**

用 Edit 工具在 `SECURITY.md` 末尾追加：

```markdown

## Audit History

This section records the periodic `cargo audit` results and any remediation taken.

### 2026-06-26 (Wave 3)

`cargo audit` baseline: 0 vulnerabilities + 3 warnings.

| Crate | Version | Advisory | Status |
|-------|---------|----------|--------|
| `openssl` | 0.10.79 | (Dependabot; no RUSTSEC) | ✅ Bumped to 0.10.80 |
| `memmap2` | 0.9.10 | RUSTSEC-2026-0186 (unsound: unchecked pointer offset) | ✅ Bumped to 0.9.11 |
| `rustls-pemfile` | 2.2.0 | RUSTSEC-2025-0134 (unmaintained) | ⚠️ Deferred — 2.x line terminated; replacement requires `crates/server/src/security/tls.rs` API refactor |
| `paste` | 1.0.15 | RUSTSEC-2024-0436 (unmaintained) | ⚠️ Deferred — deep transitive via `gemm` → `candle-core`; resolution requires candle-core upgrade (independent scope) |

Post-remediation audit: 2 warnings remaining (rustls-pemfile + paste).

Refs: `docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md`
```

- [ ] **Step 3: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat SECURITY.md
# 预期: 1 file changed, ~17 insertions

cargo check --workspace  # sanity
git add SECURITY.md
git commit -m "docs(security): add Wave 3 audit history with openssl/memmap2 fixes and deferred items"
```

---

## Task 3: SESSION-HANDOFF 更新（1 commit）

**Files:**
- Modify: `.planning/SESSION-HANDOFF.md`

- [ ] **Step 1: 更新顶部 Git 行**

Find:
```
> Git：`main` @ `b5c587e` (Wave 1 + 1.6 + 2 全部完成：18 commits；Phase 0–5 + SPEC-ADAPT counter wire-up)
```

Replace with (use `git log --oneline -1` to get actual current hash):
```
> Git：`main` @ `<Wave 3 最新 commit hash>` (Wave 1 + 1.6 + 2 + 3 全部完成：21 commits)
```

- [ ] **Step 2: 替换"下一优先级"段**

Find the section `## 下一优先级（2026-06-26，Wave 2 完成）`. Replace the entire section with:

```markdown
## 下一优先级（2026-06-26，Wave 3 完成）

**Wave 1 + 1.6 + 2 + 3 全部完成（21 commits）**

| Wave | Commit 范围 | 描述 |
|------|------------|------|
| 1 | `d42b151` ~ `1499fcd` | 文档同步 + dead_code 审计（11 commits） |
| 1.6 | `a4886a7` | 清理 vllm-model pre-existing clippy（11 lints） |
| 2 | `9e564f6` ~ `b5c587e` | SPEC-ADAPT counter wire-up + docs sync（5 commits） |
| 3 | `c93ba5e` ~ `<end>` | Dependabot bumps: openssl 0.10.80 + memmap2 0.9.11 + SECURITY.md audit history（3 commits） |

**下一 Wave:** Wave 4 (SPEC-WARM-01 speculative warmup)
- prefill draft model KV cache before decode
- 需扩展 `SelfSpeculativeModel` 或草稿模型预热路径
- 中等工作量

**Wave 3 spec/plan:**
- Spec: `docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md` (commit `c93ba5e`)
- Plan: `docs/superpowers/plans/2026-06-26-wave3-dependabot-bumps.md` (本文件)

**Deferred from Wave 3 (记录于 SECURITY.md):**
- rustls-pemfile 2.2.0 unmaintained (RUSTSEC-2025-0134): 需 tls.rs API 重构
- paste 1.0.15 unmaintained (RUSTSEC-2024-0436): deep transitive via gemm→candle；需 candle 升级
```

(用 `git log --oneline` 确认 hash 后填入)

- [ ] **Step 3: 验证 + commit**

```bash
cd /workspace/vllm-lite
git diff --stat .planning/SESSION-HANDOFF.md
# 预期: 1 file changed, ~25 insertions, ~10 deletions

cargo check --workspace  # sanity
git add .planning/SESSION-HANDOFF.md
git commit -m "docs(planning): refresh SESSION-HANDOFF for Wave 3 status"
```

---

## 收口验证

所有 4 commits 完成后（spec 已 + 3 new = 4）：

```bash
cd /workspace/vllm-lite

# 1. 全量 CI
just ci

# 2. Cargo.lock diff 范围
git diff origin/main HEAD -- Cargo.lock | grep -E "^[+-]" | grep -E "openssl|memmap2"
# 预期: 仅 openssl + openssl-sys + memmap2 相关行有变更

# 3. Audit 状态（应从 3 warnings 降到 2）
cargo audit 2>&1 | grep -c "^Crate:"
# 预期: 2（rustls-pemfile + paste）

# 4. 文档一致性
rg "Wave 3" SECURITY.md
# 预期: 反映完成态

rg "Wave 3" .planning/SESSION-HANDOFF.md
# 预期: 反映完成态

# 5. 测试基线
just nextest 2>&1 | tail -3
# 预期: ≥ 1035 passed
```

**Wave 3 完成标志：**
- ✅ `just ci` 全绿
- ✅ `just nextest` ≥ 1035 passed（无回归）
- ✅ `cargo audit` warnings 从 3 降到 2
- ✅ `Cargo.lock` diff 仅 openssl + memmap2
- ✅ `SECURITY.md` 有 Wave 3 audit history 段
- ✅ `SESSION-HANDOFF` 反映 Wave 3 完成

---

## 错误处理 / 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| memmap2 0.9.11 与 candle-core ABI 不兼容 | 低-中 | 中 | Task 1 Step 4 build 失败 → revert memmap2，保留 openssl；记入 SECURITY.md |
| `cargo update -p memmap2` 拉入其他 deps 间接变更 | 中 | 低 | Step 7 diff 检查；若无关变更则 revert 重做 |
| openssl 系统库版本不兼容 | 极低 | 中 | CI 环境 OpenSSL ≥ 1.0.1 即可 |
| Audit 重新跑发现其他隐藏 issue | 中 | 低 | 评估 + 记录；不动范围内则 deferred |

---

## 自审

- **Spec 覆盖:** ✅ D3-1 (openssl bump) → Task 1；D3-2 (memmap2 bump) → Task 1；D3-3 (rustls-pemfile deferred) → Task 2；D3-4 (paste deferred) → Task 2；D3-5 (SECURITY.md audit history) → Task 2；SESSION-HANDOFF → Task 3
- **占位符扫描:** ✅ 无 TBD/TODO；每步有具体命令和预期
- **类型一致性:** ✅ N/A（无代码类型变更；仅 Cargo.lock + docs）
- **范围:** ✅ 3 commits（spec 已 1 + 3 new），单次会话可完成

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 `docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md` |
