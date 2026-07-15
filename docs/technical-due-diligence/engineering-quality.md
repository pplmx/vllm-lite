# 工程质量、测试、构建与依赖评估

## 1. 总体判断

工程基础约为 **7/10**：Rust workspace、lint、nextest、property/fuzz/mutation testing、
API baseline 和供应链检查均有较好投入。主要问题是本地命令、CI、Docker 与真正产品
路径并不一致，导致“检查很多”但 GPU 推理核心风险仍可能漏过。

## 2. 可验证规模

- 6 个 workspace crate；`fuzz` 为独立 workspace。
- `crates/` 下约 477 个 Rust 文件。
- 约 56 个集成测试文件，README 宣称 1235 个 passing tests。
- 约 41 个 ignored 慢测试/真实 checkpoint 测试。
- 12 个 Criterion benchmark。
- 8 个 fuzz target，PR/nightly 仅覆盖其中部分。
- Rust 1.88、Edition 2024；根 `Cargo.lock` 已提交。

精确数字会随提交变化，README 不应手工维护具体测试数量。

## 3. CI-01：本地与 CI feature 矩阵漂移

**证据**

- `.github/workflows/ci.yml:37-53` 的 clippy/nextest 不使用 `--all-features`。
- `justfile:17-18,39-45` 的 canonical 本地命令使用 `--all-features`。
- `CONTRIBUTING.md` 声称 `just ci` 与 GitHub Actions 一致。

**根因**

为适配无 GPU 的 GitHub runner，CI 去掉了全 feature，但文档和本地入口未同步，也没有
建立 “full-minus-cuda” 等明确组合。

**影响**

PR 可在 CI 通过、却在维护者本地失败；`multi-node`、`cuda-graph`、`gguf` 等组合的
编译回归不能稳定发现。

**严重性 / 优先级**：High / P1  
**复杂度**：低–中  
**收益**：高

**方案**

1. 建立 `default`、`full-minus-cuda`、`multi-node` 三层矩阵；信号清晰但增加 CI 时间。
2. 增加仅编译的 `--all-features` job；成本低，但运行时覆盖仍有限。
3. 将 `just ci` 降为默认 features；一致性提高，但会主动降低本地质量门槛，不推荐。

## 4. 测试策略

### 优势

- 大量模块内 unit test 与 crate integration test。
- nextest 有 optimized/checkpoint profile 和 timeout 管理。
- scheduler/sampling 使用 proptest。
- loader/config/HTTP parser 有 fuzz 基础。
- core engine 有 mutation testing 基线。
- doctest 与 public API diff 已进入 CI。

### 结构性盲区

1. 真实 checkpoint、CUDA Graph 和 GPU 模型测试默认 ignored。
2. PR 仅 Linux CPU；Windows/macOS 只在 release 构建，缺少测试。
3. proptest 主要集中在 core，loader、server、dist 覆盖不足。
4. mutation testing 未进入 CI，且依赖 `--baseline skip` 绕过已知失败。
5. 行覆盖率只有本地 tarpaulin 命令，无趋势或门禁。
6. 安全模块多为隔离 router 测试，没有覆盖真实 `main` 组装结果。

**根因**

测试规划按技术方法扩展，而不是按产品风险矩阵组织；GPU runner、模型权重与运行成本
也阻碍了持续执行。

**建议测试金字塔**

- 每个 PR：fmt、clippy、doc、default/full-minus-cuda、CPU unit/integration、
  OpenAI contract、router composition。
- 每晚：全部 fuzz target、ignored 非 GPU 测试、跨平台 check。
- 每周或 release candidate：固定小模型 checkpoint、单 GPU 端到端、长上下文、
  并发/取消/过载、性能基线。
- 发布前：支持模型矩阵、容器启动、Helm render/smoke、升级与回滚演练。

测试指标应从“测试数量”转向：关键场景覆盖率、flaky rate、变异分数、GPU 基准漂移和
从失败到修复的时间。

## 5. Benchmark 与性能治理

**证据**

- `crates/core/benches` 和 `crates/model/benches` 已有 12 个 benchmark。
- `.github/workflows/benchmark.yml` 与 `performance-regression.yml` 仍声称没有
  benchmark target 或只是 placeholder。
- README 的吞吐、TTFT、P99、显存收益未链接具体 benchmark artifact。

**影响**

性能声明不可复现；优化提交无法区分真实收益、测量噪声和硬件变化。

**方案**

1. 立即启用 CPU `radix_cache` 等稳定微基准，只做趋势告警而非硬门禁。
2. 自建固定 GPU runner，持久化环境指纹和 Criterion/自定义 serving 结果。
3. 为 README 只展示最近稳定 release 的结果，并链接完整方法、原始数据和 commit SHA。

## 6. Feature 与依赖管理

### Feature 问题

- `traits` 的 `kernels` 是空 feature。
- `server` 硬编码启用 core 的 `cuda-graph`，不利于最小部署。
- `dist` 不在 default-members，分布式回归容易漏检。
- 各 crate 的 `cuda`、`full`、`multi-node` 传播缺少统一用户模型。

建议建立文档化 feature matrix，明确 CPU reference、CUDA、GGUF、distributed 的合法
组合；无实际 gate 的 feature 应删除或真正实施。

### §6 闭合状态（v31.0 P15）

逐项核对四条 Feature 问题，结论是 **3 项已实质关闭 + 1 项半关闭**，均不阻塞 v31.0 alpha：

| # | 原问题 | 当前状态 | 关闭依据 |
|---|--------|----------|----------|
| 1 | `traits` 的 `kernels` 是空 feature | ✅ 已关闭（误判） | `kernels = []` 是 Cargo 的"feature 存在但不引入新依赖"语法；`crates/traits/src/lib.rs:25-26` 的 `#[cfg(feature = "kernels")]` 实际门控 `kernels` 模块（`CudaGraphConfig` / `CudaGraphExecutor` / `GraphExecutionError` / `ModelGraphConfig`）。Feature 既是 cfg 门又有真实代码，闭合。 |
| 2 | `server` 硬编码启用 core 的 `cuda-graph` | ✅ 已关闭 | `crates/server/Cargo.toml` 显式 `vllm-core = { path = "../core", default-features = false }`，且 `cuda-graph = ["vllm-core/cuda-graph"]` 是**非默认** feature。最小部署开箱即用，不引入 cuda-graph。 |
| 3 | `dist` 不在 default-members | ✅ 已关闭 | `Cargo.toml` workspace 段：`default-members = ["crates/core", "crates/model", "crates/server", "crates/traits", "crates/dist", "crates/testing"]` —— 6 个 crate 全部在 default-members。注意：ADR-008 的内容曾建议 `dist` **不**在 default-members（与早期 v19.x 状态一致），但 v31.0 期间的多节点投资决策（ADR-015 / Phase 31-D OPS-31d）将其提升回 default-members，因为现在 `dist` 有真实消费方。 |
| 4 | 各 crate 的 `cuda`、`full`、`multi-node` 传播缺少统一用户模型 | 🟡 半关闭 | `docs/architecture.md` §"Feature Flags" 给出了当前合法组合的总结；但完整的"feature matrix 文档"（明确 CPU reference / CUDA / GGUF / distributed 的合法组合）尚未写。当前 `ARCHITECTURE.md` 是事实来源。建议保留为 v0.2 / 32+ 的"feature matrix doc" 工作项。 |

**净结论：** 第 1-3 项是 due diligence 撰写时（v19.x 阶段）与 v31.0 workspace 实际状态脱节的 stale 关注点。第 4 项是真实的 follow-up，已通过 `docs/architecture.md` 部分覆盖，但完整 feature matrix 仍是 v0.2/32+ 工作。

### 依赖问题

- 根 lock 中存在多版 rand、safetensors 等依赖。
- `multiple_crate_versions` 在 clippy 被 allow，在 cargo-deny 仅 warn。
- `fuzz/Cargo.toml` 的 MSRV 和解析依赖版本落后于主 workspace。
- 没有 Dependabot/Renovate。

重复版本部分来自 Candle/tokenizers 的传递依赖，不能机械清零。应先对安全、二进制
体积和编译时间有实质影响的重复项设置 deny/skip 说明，再引入 weekly grouped updates。

## 7. MSRV 与可复现构建

**证据**

- 根 `Cargo.toml` 要求 Rust 1.88。
- Dockerfile builder 使用 Rust 1.82。
- fuzz workspace 声明 Rust 1.85。
- 仓库无 `rust-toolchain.toml`。
- Docker build 未统一使用 `--locked`。

**严重性 / 优先级**：High / P0–P1  
**复杂度**：低  
**收益**：高

建议：

1. 统一 workspace、fuzz、Docker、教程和 CI 到 1.88。
2. 添加 `rust-toolchain.toml`，明确 components。
3. 所有 release/container build 使用 `--locked`。
4. release 生成 SBOM、校验和与构建 provenance；air-gapped vendor 按真实用户需求再做。

### §7 闭合状态（v31.0 P15）

逐项核对四条证据 + 四条建议，结论是 **4 项全部关闭**（DEP-01 / P11 / P13），仅第 4 条的"checksums + provenance"半边仍 tracked：

| # | 原问题 / 建议 | 关闭依据 | 关闭批次 |
|---|--------------|----------|----------|
| 1 | 统一 workspace、fuzz、Docker、教程和 CI 到 1.88 | 实际：根 `Cargo.toml` `rust-version = "1.88"` + `rust-toolchain.toml` `channel = "1.88"` + `fuzz/Cargo.toml` `rust-version = "1.88"` + `Dockerfile` `FROM rust:1.88-bookworm` + CI `dtolnay/rust-toolchain@1.88` | DEP-01 |
| 2 | 添加 `rust-toolchain.toml`，明确 components | `rust-toolchain.toml` 文件存在，`channel = "1.88"` 明确 | DEP-01 |
| 3 | 所有 release/container build 使用 `--locked` | `release.yml` build job `cargo build --release ... --locked`；Dockerfile 使用 `--locked`（DEP-01 关闭） | DEP-01 |
| 4a | release 生成 SBOM | `.github/workflows/release.yml` 的 `anchore/sbom-action@v0` 步骤，每 target 输出 cyclonedx-json SBOM | P11 |
| 4b | release 生成校验和 | **未关闭** —— 仍 tracked 为 v32+ candidate；见 STATE.md "Remaining open items (after P14) — engineering-quality §7 checksums + provenance" | (P12+) |
| 4c | release 生成构建 provenance (SLSA / in-toto) | **未关闭** —— 与 4b 同组；需要签名密钥故事 + 可复现构建姿态，超出单 CI step 范围 | (P12+) |

**净结论：** §7 的 4 条建议中 3 条实质关闭、1 条半关闭。SBOM（P11）已 ship；checksums + provenance 是 v32+ 候选。本节不阻塞 v31.0 alpha。

## 8. 开发者体验

优点是 `justfile` 覆盖常见任务，CONTRIBUTING 详细，错误类型和 crate 根 re-export
有统一规范。问题包括：

- `just init` 和部分 recipe 依赖 uv、prek、bash/sh，Windows 原生体验一般。
- 多个“canonical”命令语义不一致。
- 文档入口很多，历史 planning/spec 文件数量巨大，搜索噪声高。

建议只保留三条首要入口：

```text
just check        快速、只读、本地循环
just ci           与 PR 阻断门禁一致
just ci-full      全 feature、安全、慢测试
```

为 Windows 提供 PowerShell 等价脚本，或明确 WSL 是唯一支持的开发路径。历史设计文档
应归档并在权威文档中避免被普通搜索误认为当前行为。

## 9. 工程质量优先事项

1. 统一 toolchain、Docker 和 `--locked`。
2. 对齐本地/CI feature 矩阵。
3. 修复 CUDA Graph 已知基线失败，取消 mutation workaround。
4. 恢复 benchmark workflow，并让 README 性能声明可追溯。
5. 建立 nightly/weekly GPU checkpoint 验证。
6. 增加真实 router composition、取消、过载与部署 smoke test。
7. 最后再提高覆盖率百分比；不要用覆盖率替代风险测试。
