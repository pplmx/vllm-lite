# Wave 3: Dependabot 漏洞修复设计

**日期**: 2026-06-26
**状态**: 🔄 待审
**基线**: `main @ 4a2d279`（Wave 1 + 1.6 + 2 全部完成）
**关联**: `.planning/PROJECT.md`；SESSION-HANDOFF "Dependabot 5 漏洞" 项

---

## 背景

### 现状（2026-06-26 `cargo audit` 实测）

| Crate | Version | Advisory | Severity | Status | 可 action |
|-------|---------|----------|----------|--------|----------|
| **openssl** | 0.10.79 | （无 RUSTSEC；Dependabot flag）| — | upstream 0.10.80 可用 | ✅ bump |
| **memmap2** | 0.9.10 | RUSTSEC-2026-0186 | unsound | upstream 0.9.11 可用 | ✅ bump |
| **rustls-pemfile** | 2.2.0 | RUSTSEC-2025-0134 | unmaintained | 2.x line 终止 | ❌ 需重构 |
| **paste** | 1.0.15 | RUSTSEC-2024-0436 | unmaintained | deep transitive via gemm→candle | ❌ 需 candle 升级 |

`cargo audit` 实际输出：**0 vulnerabilities + 3 warnings**（memmap2 unsound + 2 unmaintained）。SESSION-HANDOFF 写的"5 vulnerabilities (1 high, 4 moderate)"在 cargo audit 视角下没有 5 个——可能是 Dependabot 的 GitHub-side 分类（reachability、metadata）使其显示为 5 alerts。

### 已存在但需重打的 Dependabot PR

`origin/dependabot/cargo/openssl-0.10.80` 分支基于 `7e8bd3a`（**Wave 1 之前**），与当前 main 差 **9497 行删除 + 6032 行新增**。**不能 merge**，需在 `main @ 4a2d279` 重新打 openssl 0.10.80 bump commit。

---

## 目标

1. **关闭可 action 的安全告警**：openssl + memmap2 两个 bump
2. **记录不可 action 的告警**：rustls-pemfile + paste 的处置策略与后续路径
3. **同步文档**：SECURITY.md / SESSION-HANDOFF 反映 audit 现状与决策

**非目标：**

- 不升级 Rust toolchain
- 不升级 candle-core（虽然其升级可能最终解决 paste + memmap2 两项）
- 不替换 rustls-pemfile 为自定义 PEM parser（侵入大，独立 spec）
- 不引入新依赖
- 不动 cargo features（避免连锁）

---

## 设计

### D3-1：openssl 0.10.79 → 0.10.80 bump

**决策：** 用 `cargo update -p openssl --precise 0.10.80` 在当前 main 上 bump。

**理由：**
- Dependabot 远程分支已验证此 bump 安全（无 API 变更）
- 仅 Cargo.lock 改动（2 行：openssl + openssl-sys）
- transitive chain：`vllm-core → metrics-exporter-prometheus → hyper-tls → native-tls → openssl`

**操作：**

```bash
cargo update -p openssl --precise 0.10.80
# 等价于：把 Cargo.lock 中的 openssl 0.10.79 → 0.10.80（openssl-sys 同步 0.9.115 → 0.9.116）
```

### D3-2：memmap2 0.9.10 → 0.9.11 bump

**决策：** 用 `cargo update -p memmap2`（自动选 0.9.11 latest）。

**理由：**
- RUSTSEC-2026-0186（unchecked pointer offset）通常在 minor patch 修复
- 仅 Cargo.lock 改动（memmap2 单行版本）
- transitive chain：`vllm-model/dist/server → candle-core → memmap2`
- candle-core 0.10.2 依赖 memmap2 范围允许 0.9.x（cargo update 会自动选）

**风险：** candle-core 0.10.2 与 memmap2 0.9.11 ABI 兼容性。如 build/test 失败，需 revert 并评估。

### D3-3：rustls-pemfile 不动（document-only）

**决策：** 不在 Wave 3 bump；记录决策 + 后续路径。

**理由：**
- 2.x line 是终止状态（rustls org 转向 rustls crate 内置 PEM 解析）
- 替换方案（如 `rustls::pki_types::PrivateKeyDer`）需 API 重构 `crates/server/src/security/tls.rs`
- 风险与工作量超出 Wave 3 scope

**文档化：** 在 SECURITY.md 添加 known issue 段；引用 RUSTSEC-2025-0134；说明替换路径。

### D3-4：paste 不动（deep transitive）

**决策：** 不在 Wave 3 bump；记录决策。

**理由：**
- paste 是 proc-macro crate，被 gemm 依赖，gemm 被 candle-core 依赖
- 提升到不 unmaintained 的 paste 替代品需先升级 candle-core
- candle-core 0.10.2 → 更新版本是独立的大版本升级

**文档化：** SECURITY.md 标注 "transitive via gemm→candle-core; 待 candle 升级时一并解决"。

### D3-5：SECURITY.md 补 audit 记录段

**决策：** 在现有 SECURITY.md 添加 `## Audit History` 段，记录：

```markdown
## Audit History

### 2026-06-26 (Wave 3)

`cargo audit` 实测结果：0 vulnerabilities + 3 warnings。

| Crate | Advisory | 状态 |
|-------|----------|------|
| openssl 0.10.79 | （无 RUSTSEC；Dependabot 标记）| ✅ bumped to 0.10.80 |
| memmap2 0.9.10 | RUSTSEC-2026-0186 unsound | ✅ bumped to 0.9.11 |
| rustls-pemfile 2.2.0 | RUSTSEC-2025-0134 unmaintained | ⚠️ deferred (待 tls.rs 重构) |
| paste 1.0.15 | RUSTSEC-2024-0436 unmaintained | ⚠️ deferred (transitive via gemm→candle) |

Refs: `docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md`
```

---

## 目标目录结构（无变化）

Wave 3 仅修改：
- `Cargo.lock`（两个版本 bump）
- `SECURITY.md`（audit history 段）

---

## 任务分解

### Wave 3 Task 1：spec doc（本文件已写）

`docs/superpowers/specs/2026-06-26-wave3-dependabot-audit.md`（本文件）

### Wave 3 Task 2：openssl + memmap2 bump（1 commit）

```bash
cd /workspace/vllm-lite
cargo update -p openssl --precise 0.10.80
cargo update -p memmap2
# 验证
cargo build --workspace
cargo build --workspace --tests
cargo clippy --workspace --all-targets -- -D warnings
just nextest
# Commit
git add Cargo.lock
git commit -m "chore(deps): bump openssl 0.10.79 -> 0.10.80 and memmap2 0.9.10 -> 0.9.11"
```

### Wave 3 Task 3：SECURITY.md 更新（1 commit）

修改 `SECURITY.md` 加 audit history 段。

### Wave 3 Task 4：SESSION-HANDOFF 更新（1 commit）

Wave 2 已 refresh 过。Wave 3 仅在 SESSION-HANDOFF "下一优先级" 段：
- 移除 "Dependabot 5 漏洞" 项
- 替换为 "Wave 3 完成" 表
- 下一 wave 改为 Wave 4 (SPEC-WARM)

---

## 验证

### Wave 3 Task 2 验证

```bash
# Bump 后 audit 重新跑
cargo audit
# 预期: warnings 数量从 3 降到 1（仅 rustls-pemfile + paste 仍在；openssl 和 memmap2 已清除）

# Build
cargo build --workspace
cargo build --workspace --tests
# 预期: 全部成功（特别是 candle-core 与 memmap2 0.9.11 ABI 兼容）

# 测试
just nextest
# 预期: ≥ 1035 passed（无回归）

# Clippy
cargo clippy --workspace --all-targets -- -D warnings
# 预期: 0 errors
```

### Wave 3 收口验证

```bash
# Cargo.lock diff
git diff origin/main HEAD -- Cargo.lock | grep -E "^[+-]" | grep -E "openssl|memmap2"
# 预期: openssl + openssl-sys + memmap2 三个版本行有变更

# Audit 状态
cargo audit 2>&1 | grep -E "^Crate|^Warning|^Title"
# 预期: 仅 rustls-pemfile + paste 仍在

# 文档
rg "Wave 3" SECURITY.md SESSION-HANDOFF.md
# 预期: 反映完成态
```

---

## 错误处理 / 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| memmap2 0.9.11 与 candle-core ABI 不兼容 | 低-中 | 中 | `cargo build` 验证；失败则 revert memmap2 bump |
| openssl-sys 0.9.116 需 OpenSSL 系统库版本 | 极低 | 中 | CI 环境若 OpenSSL ≥ 1.0.1 即可 |
| `cargo update -p memmap2` 拉入其他 deps 间接变更 | 中 | 低 | `cargo update -p X` 应只动 X 与其依赖；diff 检查 |
| audit 重跑发现其他隐藏 issue | 中 | 低 | 评估 + 记录；不动范围内则 deferred |

---

## 不做（明确边界）

- ❌ 不升级 Rust toolchain
- ❌ 不升级 candle-core（独立 spec 工作）
- ❌ 不替换 rustls-pemfile 为自定义 parser（侵入大）
- ❌ 不引入新依赖（如 rustls-pemfile 替代品）
- ❌ 不动 cargo features
- ❌ 不删依赖（即使看似未用，candle transitive 可能用）
- ❌ 不修任何代码（pure Cargo.lock + docs）

---

## 风险与决策记录

| ID | 决策 | 理由 | 日期 |
|----|------|------|------|
| D3-1 | `cargo update -p openssl --precise 0.10.80` | Dependabot 已审，patch bump 无 API 变更 | 2026-06-26 |
| D3-2 | `cargo update -p memmap2`（自动选 0.9.11） | latest patch 可能修复 RUSTSEC-2026-0186 | 2026-06-26 |
| D3-3 | rustls-pemfile 不动 | 替换需 tls.rs API 重构，超 Wave 3 scope | 2026-06-26 |
| D3-4 | paste 不动 | deep transitive via gemm→candle | 2026-06-26 |
| D3-5 | SECURITY.md 加 Audit History 段 | 持续可追溯的 audit 记录 | 2026-06-26 |

---

## 会话接续

Wave 3 完成后：
1. 读 SESSION-HANDOFF（已更新）确认 Wave 3 状态
2. `cargo audit` 重跑确认 warning 减少
3. 下一 Wave 候选：Wave 4 (SPEC-WARM-01 speculative warmup) 或其他

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-06-26 | 初版：基于 `cargo audit` 与 cargo tree 实测结果 |
