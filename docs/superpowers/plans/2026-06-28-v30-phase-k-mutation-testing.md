# v30.0 Phase K: Mutation Testing 接入

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 `cargo-mutants` 在 vllm-core 的 scheduler / sampling / radix_cache 模块运行 mutation testing,生成 mutation score baseline,处置高生存 mutation,使核心模块 mutation score ≥ 70%。

**Architecture:**
- `cargo-mutants` 作为外部工具,通过 `cargo install` 一次性安装
- 工作流:`just mutants MODULE` 触发单模块 mutation scan → 输出 `.mutants-out/mutations.json` + 人类可读 report
- 三阶段推进:K-1 baseline (空跑) → K-2 triage (处置高生存 mutation) → K-3 maintenance (脚本+文档)
- scope 限定:`crates/core/src/scheduler/**`,`crates/core/src/sampling.rs`。model 层和 server 层不在本次范围(计算密集 / IO 密集)

**Tech Stack:** `cargo-mutants` (latest stable, ≥ 25.0), Rust 1.85 stable

**Phase 关联:**
- 上游:无(v23-v29 已完成,本阶段起手)
- 下游:v30 Phase L (fuzz CI)、Phase M (coverage expansion) 会引用本阶段 mutation 报告作为对照基线

---

## File Structure

| File | Change | Sub-phase |
|------|--------|-----------|
| `justfile` | 新增 `mutants`, `mutants-report`, `mutants-clean` 三个 target | K-1, K-3 |
| `docs/testing/mutation-testing.md` | mutation testing 选型说明 + 使用方法 | K-3 |
| `.mutants-out/` | cargo-mutants 输出目录(gitignored) | K-1 |
| `CHANGELOG.md` | v30.0 Phase K 条目 | K-3 |
| `.gitignore` | 追加 `.mutants-out/` | K-1 |

不修改任何生产代码 — K 阶段只引入工具与文档。

---

## Scope Definition

### In-scope modules

```text
crates/core/src/scheduler/
├── batch.rs (99 lines)
├── batch_composer/
│   ├── compose.rs (858 lines)
│   └── validate.rs (162 lines)
├── engine/
│   ├── graph.rs (81 lines)
│   ├── memory.rs (114 lines)
│   ├── mod.rs (170 lines)
│   ├── state.rs (424 lines)
│   └── update.rs (121 lines)
├── memory/
│   ├── allocator.rs (363 lines)
│   ├── eviction.rs (313 lines)
│   └── mod.rs (258 lines)
├── policy/
│   ├── fcfs.rs (30 lines)
│   ├── priority.rs (82 lines)
│   └── sjf.rs (87 lines)
├── radix_cache/
│   ├── node.rs (45 lines)
│   └── tree.rs (213 lines)
├── request_queue.rs (438 lines)
└── phase_scheduler.rs (216 lines)

crates/core/src/sampling.rs (343 lines)
```

**Total:** ~4415 lines across 19 files

### Out-of-scope (deferred)

- `crates/core/src/scheduler/cuda_graph.rs` — 需要 `cuda-graph` feature,影响面大
- `crates/core/src/scheduler/observer.rs`、`stats.rs` — 主要是 metrics emission,mutant 价值低
- `crates/core/src/scheduler/packing.rs` — 与 model layer 强耦合,独立测试困难
- `crates/model/` — 计算密集,mutant 跑得慢
- `crates/server/` — IO 密集,mutant 收益低

---

## Audit-Driven Constraints

### 已知 mutation 难点

1. **`scheduler/radix_cache/tree.rs`** — 内部状态复杂,容易出现 "equivalent mutant"(如 `tree.match_prefix` 中 token id 比较的等价变形)
2. **`sampling.rs`** — `top_k`/`top_p` 的浮点比较,mutant 经常是 equivalent
3. **`memory/eviction.rs`** — 时序敏感的逻辑,mutant 难以触发 fail

### CI 时间预算

- 单次完整 mutation scan 预估 60-90 分钟
- K 阶段在本地执行,**不纳入 CI**(留给 v31 评估)
- 短扫描(`--timeout 30` + `--jobs 4`)可缩到 ~20 分钟,适合开发期间

---

## Sub-phase Plan

### K-1: 接入 cargo-mutants 并生成 Baseline Report (3 tasks)

**K-1.1:** 安装 cargo-mutants,验证命令可用
**K-1.2:** 添加 `.mutants-out/` 到 `.gitignore`,添加 justfile targets
**K-1.3:** 在最小模块 (`scheduler/policy/`) 跑首次 mutation,生成 baseline report

### K-2: Triage 高生存 Mutation (3 tasks)

**K-2.1:** 跑完整 scheduler 模块 mutation scan,导出 JSON
**K-2.2:** 排序 top-N 高生存 mutation,分类(equivalent / weak-test / real-bug)
**K-2.3:** 为 "weak-test" 类 mutation 各补一个针对性测试

### K-3: 维护脚本 + 文档 (2 tasks)

**K-3.1:** 编写 `just mutants MODULE` 完整工作流 + `mutants-report` / `mutants-clean` 辅助
**K-3.2:** 编写 `docs/testing/mutation-testing.md` + 更新 CHANGELOG

---

## Task K-1.1: Install cargo-mutants

**Files:**
- Modify: (none)
- Create: (none) — cargo-mutants 安装在 `~/.cargo/bin/`

- [ ] **Step 1: 确认环境**

Run:
```bash
rustc --version
cargo --version
```
Expected: Rust 1.85.0+ / cargo 1.85.0+

- [ ] **Step 2: 安装 cargo-mutants**

Run:
```bash
cargo install cargo-mutants --locked
```
Expected: 安装成功,输出 `Installed package cargo-mutants v25.x.x`

- [ ] **Step 3: 验证可用**

Run:
```bash
cargo mutants --version
```
Expected: `cargo-mutants 25.x.x`

- [ ] **Step 4: 验证 workspace 上下文**

Run:
```bash
cd /workspace/vllm-lite && cargo mutants --list-files --package vllm-core 2>&1 | head -20
```
Expected: 列出 `vllm-core` 包的文件(若有 error 提示 cargo-mutants 不支持 `--list-files`,记录 issue,继续)

---

## Task K-1.2: Add .gitignore entry and justfile targets

**Files:**
- Modify: `.gitignore`
- Modify: `justfile`

- [ ] **Step 1: 在 .gitignore 追加 .mutants-out**

Read `.gitignore` first, then add at end:

```gitignore
# cargo-mutants output (v30.0 Phase K)
.mutants-out/
```

- [ ] **Step 2: 在 justfile 追加 mutants targets**

Read `justfile` first, find a logical insertion point (after `fuzz-*` targets), then add:

```makefile
# === Mutation testing (v30.0 Phase K) ===
# Generate baseline mutation scan for an entire module under vllm-core.
# Usage: `just mutants MODULE` where MODULE is a path relative to
# crates/core/src (e.g. `scheduler`, `scheduler/policy`, `sampling`).
# NOTE: First run downloads and caches mutants tool (~minutes), subsequent
# runs reuse the cache. A full scheduler scan takes ~30-60 min on 4 cores.
mutants MODULE:
    cargo mutants \
        --package vllm-core \
        --file "crates/core/src/{{MODULE}}" \
        --timeout 30 \
        --jobs $(nproc) \
        --output .mutants-out/ \
        --shuffle

# Render a human-readable summary of the latest mutation scan.
mutants-report:
    @test -d .mutants-out || (echo "no .mutants-out/ — run `just mutants MODULE` first"; exit 1)
    @if [ -f .mutants-out/mutations.json ]; then \
        jq -r '.[] | select(.status == "SURVIVED") | "\(.file):\(.span.start_line)  \(.mutant_name)"' \
            .mutants-out/mutations.json | head -30; \
    else \
        echo "mutations.json not found; check .mutants-out/ contents"; \
    fi

# Remove the mutation output directory.
mutants-clean:
    rm -rf .mutants-out
```

- [ ] **Step 3: 验证 justfile 语法**

Run:
```bash
cd /workspace/vllm-lite && just --list 2>&1 | grep -E "mutants"
```
Expected: 输出
```text
mutants MODULE    # Generate baseline mutation scan for an entire module under vllm-core.
mutants-report    # Render a human-readable summary of the latest mutation scan.
mutants-clean     # Remove the mutation output directory.
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore justfile
git commit -m "build(mutant): add cargo-mutants justfile targets + gitignore (K-1.2)"
```

---

## Task K-1.3: Baseline scan on smallest module

**Files:**
- Create: `.mutants-out/` (gitignored — produced by tool)

- [ ] **Step 1: 在 scheduler/policy 模块跑首次 mutation**

Run (注意:可能耗时 5-15 分钟):
```bash
cd /workspace/vllm-lite && just mutants scheduler/policy 2>&1 | tee /tmp/mutants-policy-baseline.log
```
Expected: 终端输出 `Mutation testing complete`,以及 `X mutants tested, Y caught, Z missed` 形式的总结

- [ ] **Step 2: 查看输出目录**

Run:
```bash
ls -la /workspace/vllm-lite/.mutants-out/ | head -20
```
Expected: 看到 `mutations.json`、`mutants.out`、`caught.txt`、`missed.txt` 等文件

- [ ] **Step 3: 查看基线统计**

Run:
```bash
cd /workspace/vllm-lite && just mutants-report
```
Expected: 列出存活 mutations(若列表为空表示所有 mutation 都被现有测试捕获 — 极少见)

- [ ] **Step 4: 记录 baseline 到 commit message**

Run:
```bash
git log --oneline -1 -- .mutants-out/ 2>/dev/null || echo "(directory gitignored, no commit)"
# 找到 baseline 报告,手动记录到本 plan 的"实际结果"区段(后续 phase audit 用)
```

- [ ] **Step 5: Commit documentation**

Run:
```bash
git add /tmp/mutants-policy-baseline.log docs/superpowers/specs/2026-06-28-v30-test-docs-design.md 2>/dev/null || true
# baseline log 不入库,只用作 K-2 triage 的对照
echo "Baseline: scheduler/policy — see /tmp/mutants-policy-baseline.log"
```

---

## Task K-2.1: Full scheduler module mutation scan

**Files:**
- Create: `docs/testing/mutation-scheduler-baseline.md` (结果报告)

- [ ] **Step 1: 跑完整 scheduler 模块 scan**

Run (预估 30-60 分钟,可后台运行):
```bash
cd /workspace/vllm-lite && just mutants scheduler 2>&1 | tee /tmp/mutants-scheduler-baseline.log
```
Expected: `Mutation testing complete` + 总结行

- [ ] **Step 2: 解析 baseline 数字**

Run:
```bash
cd /workspace/vllm-lite && grep -E "(caught|missed|timeout|unviable)" /tmp/mutants-scheduler-baseline.log | tail -20
```
Expected: 看到类似 `X caught / Y missed / Z timeout` 的统计行

- [ ] **Step 3: 写 baseline 报告**

Create `docs/testing/mutation-scheduler-baseline.md`:

```markdown
# vllm-core scheduler Module — Mutation Baseline

**Date:** $(date -u +%Y-%m-%d)
**Tool:** cargo-mutants v25.x.x
**Scope:** `crates/core/src/scheduler/**` (excluding `cuda_graph.rs`, `observer.rs`, `stats.rs`, `packing.rs`)

## Summary

| Status | Count |
|--------|-------|
| Caught | (from log) |
| Missed (survived) | (from log) |
| Timeout | (from log) |
| Unviable | (from log) |
| **Total** | (from log) |

## Mutation Score

Score = caught / (caught + missed) = (X)%

## Top Missed Mutations

(see `.mutants-out/mutations.json` filtered by `status == "SURVIVED"`)

## Next Actions

(see Phase K-2.2 triage)
```

- [ ] **Step 4: Commit baseline 报告**

```bash
git add docs/testing/mutation-scheduler-baseline.md
git commit -m "docs(test): Phase K baseline mutation scan for scheduler (K-2.1)"
```

---

## Task K-2.2: Triage top-N surviving mutations

**Files:**
- Modify: `docs/testing/mutation-scheduler-baseline.md`

- [ ] **Step 1: 导出 missed mutations 为可读列表**

Run:
```bash
cd /workspace/vllm-lite && jq -r '.[] | select(.status == "SURVIVED") | "\(.file):\(.span.start_line)\t\(.mutant_name)\t\(.replacement)"' .mutants-out/mutations.json | sort > /tmp/missed.txt
wc -l /tmp/missed.txt
```
Expected: 显示 missed mutations 总数

- [ ] **Step 2: 按文件聚合,识别热点**

Run:
```bash
cut -f1 /tmp/missed.txt | sort | uniq -c | sort -rn | head -10
```
Expected: 列出 missed mutations 最多的 top-10 文件

- [ ] **Step 3: 抽样审查 top-5 mutations**

For each of the top 5 file:line entries:
1. Read the source file at that line
2. Determine if the mutation is:
   - **equivalent** — 逻辑等价(如 `x == y` 与 `!(x != y)` 在某些类型上等价)
   - **weak-test** — 现有测试覆盖路径未触发该 mutation
   - **real-bug** — 真正的代码 bug

- [ ] **Step 4: 在 baseline 报告追加 triage 表**

Modify `docs/testing/mutation-scheduler-baseline.md`,append:

```markdown
## Triage (Phase K-2.2)

| File:Line | Mutant | Category | Disposition |
|-----------|--------|----------|-------------|
| (filled in) | | equivalent/weak-test/real-bug | skip/fix-test/fix-code |
```

- [ ] **Step 5: Commit triage**

```bash
git add docs/testing/mutation-scheduler-baseline.md
git commit -m "docs(test): Phase K-2.2 mutation triage results (K-2.2)"
```

---

## Task K-2.3: Add targeted tests for weak-test mutations

**Files:**
- Modify: per-file in `crates/core/src/scheduler/**/tests.rs` or new test module

- [ ] **Step 1: 列出待补测试的 mutation 清单**

Run:
```bash
grep "weak-test" /workspace/vllm-lite/docs/testing/mutation-scheduler-baseline.md
```
Expected: 列出 K-2.2 标为 weak-test 的 mutation 位置

- [ ] **Step 2: 为首个 weak-test mutation 写失败测试**

For each weak-test mutation:
1. Read source code at that location
2. Identify the exact logic change the mutation introduces
3. Write a unit test in the appropriate `tests.rs` file that exercises a path distinguishing the original vs mutated logic

Example pattern (illustrative — actual code depends on triage):
```rust
#[test]
fn policy_priority_orders_strictly_descending() {
    let mut queue: PriorityQueue<Request> = PriorityQueue::new();
    queue.push(req_with_priority(5));
    queue.push(req_with_priority(10));
    queue.push(req_with_priority(7));
    let p1 = queue.pop().unwrap().priority();
    let p2 = queue.pop().unwrap().priority();
    let p3 = queue.pop().unwrap().priority();
    assert!(p1 >= p2);
    assert!(p2 >= p3);
    assert!(p1 >= 10);  // catches "<" → ">" mutation
}
```

- [ ] **Step 3: 跑测试验证**

Run:
```bash
cd /workspace/vllm-lite && cargo test -p vllm-core --test scheduler_priority_orders_strictly_descending
```
Expected: PASS

- [ ] **Step 4: 重跑 mutation 验证捕获**

Run:
```bash
cd /workspace/vllm-lite && cargo mutants \
    --package vllm-core \
    --file crates/core/src/scheduler/policy/priority.rs \
    --timeout 30 \
    --output .mutants-out-policy-retest/ \
    2>&1 | tee /tmp/mutants-priority-retest.log
```
Expected: 该 mutation 现在状态为 `caught` 而非 `survived`

- [ ] **Step 5: 对剩余每个 weak-test mutation 重复 steps 2-4**

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/scheduler/
git commit -m "test(core): add targeted tests for K-2.3 weak-test mutations"
```

---

## Task K-3.1: Refine justfile workflow

**Files:**
- Modify: `justfile`

- [ ] **Step 1: 跑 mutation score 统计脚本**

Create script logic inline in justfile. Read current justfile mutants section, then expand to:

```makefile
# === Mutation testing (v30.0 Phase K) ===

# Run a baseline mutation scan for a module.
# Usage: `just mutants MODULE`
mutants MODULE:
    cargo mutants \
        --package vllm-core \
        --file "crates/core/src/{{MODULE}}" \
        --timeout 30 \
        --jobs $(nproc) \
        --output .mutants-out/ \
        --shuffle

# Run mutation scan with a baseline comparison — fails if score regresses.
# Usage: `just mutants-ci MODULE BASELINE_FILE`
mutants-ci MODULE BASELINE:
    @test -f "{{BASELINE}}" || (echo "baseline file not found"; exit 1)
    cargo mutants \
        --package vllm-core \
        --file "crates/core/src/{{MODULE}}" \
        --timeout 30 \
        --jobs $(nproc) \
        --output .mutants-out/ \
        --shuffle
    @./scripts/check_mutation_score.sh .mutants-out/ {{BASELINE}}

# Render human-readable summary of latest scan.
mutants-report:
    @test -d .mutants-out || (echo "no .mutants-out/ — run \`just mutants MODULE\` first"; exit 1)
    @if [ -f .mutants-out/mutations.json ]; then \
        jq -r '.[] | select(.status == "SURVIVED") | "\(.file):\(.span.start_line)  \(.mutant_name)"' \
            .mutants-out/mutations.json | head -30; \
    else \
        echo "mutations.json not found; check .mutants-out/ contents"; \
    fi

# Print mutation score (caught / (caught+missed) %).
mutants-score:
    @test -f .mutants-out/mutations.json || (echo "no scan yet"; exit 1)
    @jq -r 'group_by(.status) | map({status: .[0].status, count: length}) | .[] | "\(.status)\t\(.count)"' \
        .mutants-out/mutations.json

# Remove the mutation output directory.
mutants-clean:
    rm -rf .mutants-out .mutants-out-*
```

- [ ] **Step 2: 写 check_mutation_score.sh 脚本**

Create `scripts/check_mutation_score.sh`:

```bash
#!/usr/bin/env bash
# scripts/check_mutation_score.sh — fail if mutation score regresses vs baseline
# Usage: scripts/check_mutation_score.sh <scan-dir> <baseline-file>
#
# baseline-file format: a single number, the minimum acceptable mutation score in % (e.g. 70)

set -euo pipefail

SCAN_DIR="${1:?usage: $0 <scan-dir> <baseline-file>}"
BASELINE_FILE="${2:?usage: $0 <scan-dir> <baseline-file>}"

MIN_SCORE=$(cat "$BASELINE_FILE")

CAUGHT=$(jq '[.[] | select(.status == "CAUGHT")] | length' "$SCAN_DIR/mutations.json")
MISSED=$(jq '[.[] | select(.status == "SURVIVED")] | length' "$SCAN_DIR/mutations.json")

if [ "$((CAUGHT + MISSED))" -eq 0 ]; then
    echo "no mutations found in scan — check scope"
    exit 2
fi

SCORE=$(awk "BEGIN { printf \"%.1f\", $CAUGHT * 100 / ($CAUGHT + $MISSED) }")

echo "mutation score: ${SCORE}% (caught=$CAUGHT, missed=$MISSED)"
echo "minimum required: ${MIN_SCORE}%"

if awk "BEGIN { exit !($SCORE < $MIN_SCORE) }"; then
    echo "FAIL: mutation score below baseline"
    exit 1
fi

echo "PASS"
```

- [ ] **Step 3: 验证 justfile + 脚本**

Run:
```bash
chmod +x /workspace/vllm-lite/scripts/check_mutation_score.sh
cd /workspace/vllm-lite && just --list 2>&1 | grep mutants
```
Expected: 列出 4 个 mutants target

- [ ] **Step 4: 在小模块 dry-run 检查**

Run:
```bash
cd /workspace/vllm-lite && bash scripts/check_mutation_score.sh .mutants-out/ <(echo 0)
```
Expected: PASS(mutation score ≥ 0%)

- [ ] **Step 5: Commit**

```bash
git add justfile scripts/check_mutation_score.sh
git commit -m "build(mutant): add mutants-ci workflow + score check script (K-3.1)"
```

---

## Task K-3.2: Document mutation testing + update CHANGELOG

**Files:**
- Create: `docs/testing/mutation-testing.md`
- Modify: `CHANGELOG.md`
- Modify: `.planning/STATE.md`

- [ ] **Step 1: 创建 docs/testing/ 目录**

Run:
```bash
mkdir -p /workspace/vllm-lite/docs/testing
```

- [ ] **Step 2: 写 mutation-testing.md**

Create `docs/testing/mutation-testing.md`:

```markdown
# Mutation Testing (v30.0 Phase K)

## Why

Property-based tests (v28.0) and fuzz tests (v29.0) validate that **known
invariants hold** for arbitrary inputs. They do not validate that tests
**fail when logic changes**. Mutation testing inverts the question:
"if a developer accidentally introduces a bug, do the existing tests catch it?"

A mutation testing tool (cargo-mutants) systematically applies small
syntactic changes ("mutations") to production code and re-runs the test
suite. A test that passes against both the original and mutated code is
"weak" — it does not actually validate the mutated logic.

## Scope (v30.0 K)

| Module | LOC | Rationale |
|--------|-----|-----------|
| `scheduler/engine/*` | ~910 | 核心调度状态机 |
| `scheduler/batch_composer/*` | ~1020 | 批组合逻辑 |
| `scheduler/memory/*` | ~934 | block 分配/驱逐 |
| `scheduler/policy/*` | ~199 | 调度策略 |
| `scheduler/radix_cache/*` | ~258 | 前缀缓存 |
| `scheduler/request_queue.rs` | 438 | 等待队列 |
| `scheduler/phase_scheduler.rs` | 216 | 阶段切换 |
| `sampling.rs` | 343 | 采样策略 |

**Excluded** in v30: `cuda_graph.rs`, `observer.rs`, `stats.rs`,
`packing.rs`, all of `model/` and `server/`. See design doc for rationale.

## Usage

```bash
# Install once
cargo install cargo-mutants --locked

# Run a baseline scan on a module (~5-60 min depending on size)
just mutants scheduler/policy

# Print summary of surviving mutations
just mutants-report

# Print mutation score
just mutants-score

# Run scan with regression check vs baseline file
just mutants-ci scheduler/policy docs/testing/mutation-baseline-policy.txt

# Clean output
just mutants-clean
```

## Baseline

See [`mutation-scheduler-baseline.md`](./mutation-scheduler-baseline.md)
for the v30.0 Phase K baseline numbers. Update after each Phase K-2.x
triage cycle.

## CI Integration

**Deferred to v31.** Mutation testing is not yet run in CI due to the
~30-60 minute scan time per module. Local runs are the source of truth
in v30.0.

## See also

- Design doc: `docs/superpowers/specs/2026-06-28-v30-test-docs-design.md` §Phase K
- ADR (forthcoming): `docs/adr/ADR-018-mutation-testing.md`
- Tool: <https://github.com/sourcefrog/cargo-mutants>
```

- [ ] **Step 3: 更新 CHANGELOG.md**

Read current CHANGELOG.md, find `[Unreleased]` section, append under "Added":

```markdown
- **Mutation Testing (v30.0 Phase K)** — cargo-mutants infrastructure for vllm-core:
    - `cargo-mutants` installed as standalone tool (no Cargo.toml change)
    - justfile targets: `mutants MODULE`, `mutants-report`, `mutants-score`, `mutants-ci MODULE BASELINE`, `mutants-clean`
    - `scripts/check_mutation_score.sh` regression checker
    - Baseline mutation scan: `crates/core/src/scheduler/**` + `crates/core/src/sampling.rs` (see `docs/testing/mutation-scheduler-baseline.md`)
    - `docs/testing/mutation-testing.md` methodology + scope
    - Mutation score target: ≥70% on core modules
    - CI integration deferred to v31 (scan time too long for current CI budget)
    - Total commits: 5 (K-1.1, K-1.2, K-1.3, K-2.1, K-2.2, K-2.3, K-3.1, K-3.2)
```

- [ ] **Step 4: 更新 .planning/STATE.md**

Read `.planning/STATE.md`, update "Current Position" and "Phase" sections to reflect v30.0 Phase K completion. Specifically:

- `milestone: v30.0`
- `milestone_name: Test Ecosystem & Docs Enhancement`
- `status: in_progress`
- `progress.total_phases: 6`
- `progress.completed_phases: 1` (Phase K)
- `progress.percent: 17`

Add to "Performance Metrics":

```markdown
**v30.0 Phase K Outcomes:**

- cargo-mutants installed and configured
- Mutation baseline established for scheduler + sampling
- Top-N mutations triaged and disposed
- Test count: (current)
```

- [ ] **Step 5: Commit**

```bash
git add docs/testing/mutation-testing.md CHANGELOG.md .planning/STATE.md
git commit -m "docs(v30.0): Phase K complete — mutation testing wired in"
```

---

## Verification

After all K tasks complete, verify:

- [ ] `just --list | grep mutants` shows 5 targets
- [ ] `just mutants scheduler/policy` runs to completion without error
- [ ] `just mutants-score` prints mutation score
- [ ] `just mutants-ci scheduler/policy docs/testing/mutation-baseline-policy.txt` passes against baseline
- [ ] `docs/testing/mutation-scheduler-baseline.md` documents triage decisions
- [ ] `docs/testing/mutation-testing.md` is readable and accurate
- [ ] CHANGELOG has v30.0 Phase K entry
- [ ] `.planning/STATE.md` reflects v30.0 K complete
- [ ] `just ci` still green (no regression to fmt/clippy/doc/nextest)

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|------------------|------|
| Install cargo-mutants | K-1.1 |
| Generate baseline report | K-1.3, K-2.1 |
| Dispose top-N high-survival mutations | K-2.2, K-2.3 |
| `just mutants MODULE` workflow | K-1.2, K-3.1 |
| Documentation (mutation testing methodology) | K-3.2 |
| Mutation score ≥70% on core modules | K-2.3 (target) |
| Equivalent mutants explicitly annotated | K-2.2 (triage) |

### Placeholder scan

No "TBD" / "TODO" / "implement later" markers introduced. All steps contain
actual commands or code. K-2.3 example test is illustrative — actual test
content depends on triage results from K-2.2.

### Type consistency

All mutations interface via `.mutants-out/mutations.json` (cargo-mutants'
standard output). All justfile targets consume this same path. The
`check_mutation_score.sh` script reads the same JSON. No type drift
across tasks.

### Risks acknowledged

- **Scan time**: K-2.1 full scheduler scan may exceed 1 hour. Mitigated
  by `--timeout 30 --jobs $(nproc)`.
- **Equivalent mutants**: Some mutations may be logical equivalents.
  K-2.2 triage categorizes them explicitly so they're not noise in
  future scans.
- **K-2.3 test additions may surface real bugs**: If a weak-test
  mutation points to a real bug, the fix may exceed K's scope. In that
  case, file as follow-up and document in baseline report.
