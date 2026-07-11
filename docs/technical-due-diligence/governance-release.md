# 文档、发布与社区治理评估

## 1. 总体判断

项目拥有 README、教程、OPERATIONS、CONTRIBUTING、SECURITY、ADR、issue/PR 模板、
CHANGELOG 和详细 planning 文档，文档数量与治理意识高于多数早期项目。主要问题是
“丰富但不同步”：新用户路径、部署路径和版本语义存在硬冲突。

## 2. 文档准确性

### 快速开始不可执行

- `README.md:72-73` 声称无参数启动并使用默认模型。
- `crates/server/src/cli/args.rs:130-133` 要求 `--model`/`VLLM_MODEL`。
- README 还声称首次运行自动下载，但实际启动路径没有相应承诺的完整实现。

**严重性 / 优先级**：Critical / P0  
**复杂度**：低  
**收益**：极高

应把 clone → build → 指定本地模型 → curl 写成 CI 可验证的单一路径。若没有模型自动
下载，就删除该声明；不要为改善文档而仓促引入网络下载与许可证复杂度。

### 性能声明不可追溯

README 声称约 2000 tokens/s、TTFT <50 ms、P99 <100 ms、显存效率 +40%，但未链接
设备、模型、commit、命令和原始结果。仓库虽有 benchmark artifacts，不能证明这些
数字来自当前实现。

建议在基准未复现前删除精确值或明确标注历史/实验条件；长期由 release benchmark
自动生成表格。

### 教程与路径漂移

- Tutorial 01 写 Rust 1.85，实际 MSRV 为 1.88。
- Tutorial 05 引用不存在的 ingress、错误扩展名、已移除的 OTel feature 和无对应 tag
  的 rollback 版本。
- 根目录没有文档所引用的可直接使用 `config.yaml`。

可执行文档应纳入链接检查、命令 smoke、compose dry-run 和 tutorial e2e。

## 3. GOV-01：版本体系冲突

**事实**

- Cargo workspace：`0.1.0`。
- Helm chart/appVersion：`0.1.0`。
- 内部 planning：v31.0。
- CHANGELOG 包含 v18–v22 等里程碑。

**根因**

内部工程里程碑编号被同时用作外部版本语义，但没有同步 Cargo、tag、镜像和迁移文档。

**影响**

用户无法判断代码、文档、镜像和 CHANGELOG 的对应关系；回滚命令和 release automation
缺少可靠输入；依赖方无法使用 SemVer 判断兼容性。

**严重性 / 优先级**：High / P1  
**复杂度**：中  
**收益**：高

**推荐**

- 内部 milestone 保留在 `.planning/`，不再进入用户版本号。
- 对外采用 SemVer 0.x；workspace version、tag、GitHub Release、镜像 tag、Chart
  appVersion 和 CHANGELOG 由单一 release manifest 驱动。
- 每次 breaking API 变化更新 MIGRATING；0.x 阶段也要说明兼容策略。

## 4. 发布流水线

`.github/workflows/release.yml` 注释承诺跨平台 binary、multi-arch GHCR 和 git-cliff
release，实际没有 Docker/GHCR job，CHANGELOG 校验也不完整。

**方案**

1. 最小发布：只发布校验和 binary，并删除未兑现承诺。简单诚实。
2. 完整发布：tag 校验 workspace version，构建三平台 binary、amd64/arm64 image，
   生成 SBOM/provenance，创建 GitHub Release，更新 Chart。
3. crates.io：若项目只作为应用，所有 crate 显式 `publish = false`；若要发布 traits，
   先缩小 API、补 metadata、建立发布顺序和 semver check。

发布候选必须复用与 PR 相同的源构建，并使用 `--locked`。不要让 release workflow 成为
第一次验证 Docker 和跨平台编译的地方。

## 5. 开源治理基础

### 优势

- MIT license 与 Cargo metadata 一致。
- CONTRIBUTING 覆盖环境、命令、测试布局和 commit 规范。
- SECURITY 记录支持版本、响应目标和依赖漏洞处置。
- bug/feature YAML 模板与 PR checklist 结构较完整。
- CODEOWNERS 对关键目录有明确所有者。
- ADR 和 DOC-MAP 为架构决策与文档权威提供基础。

### 缺口

- CODE_OF_CONDUCT 仍含 `[INSERT CONTACT EMAIL]`。
- SECURITY 没有明确 private advisory/邮箱入口。
- bug/PR 模板示例端口为 8080，服务默认是 8000。
- 无 Dependabot/Renovate。
- 无明确 MAINTAINERS、SUPPORT、release owner 和 deprecation policy。

这些多为低复杂度修复，适合 1–2 周内完成。

## 6. Bus factor 与社区健康

本地可见 CODEOWNERS 均指向单一账号，有限 Git 历史也以单一作者高频提交为主。因此可
谨慎判断 bus factor 接近 1；由于未读取远程 GitHub 数据，不能据此断言社区无贡献者。

**风险**

- 关键安全、发布和架构知识集中。
- 高频大批次变更增加审查盲区。
- 外部贡献者难以判断哪些 planning 文档仍有效。

**建议**

1. 添加 `MAINTAINERS.md`，明确领域 owner、review 要求和 release backup。
2. 将 roadmap 转成可领取、边界清楚的 issue；提供 good first issue 与架构导读。
3. 对 core/model/server 分别培养至少两名 reviewer。
4. 发布节奏固定化，减少内部 milestone 大爆发。
5. 对安全报告和行为准则提供真实、私密、可持续的联系方式。

## 7. 公共 API 与长期兼容

项目约有 6,000+ public items，文档覆盖已有改善，但公开面仍远大于早期项目通常需要的
稳定表面。过大的 API 会使重构 scheduler、sampling 和 kernel 变得昂贵。

建议把 API 分三层：

- Stable：crate root 的少量用户入口，遵循 SemVer。
- Experimental：feature gate 或 `experimental` module，可更快演进。
- Internal：`pub(crate)`，不承担兼容义务。

保留 cargo-public-api baseline，但规则应检查“breaking change 是否有迁移说明”，
而不是阻止合理收缩。文档覆盖目标应优先 Stable API。

## 8. 文档信息架构

建议收敛成：

```text
README                 可执行的 10 分钟路径与诚实能力矩阵
docs/architecture.md   当前架构唯一真相源
docs/user-guide/       安装、模型、API、部署
docs/contributor/      开发、测试、发布、ADR
docs/reference/        配置、feature、兼容矩阵
docs/archive/          历史 spec/plan，不参与当前行为说明
```

`.planning` 和大量 superpowers 文档可以保留历史价值，但应明确 archival，避免搜索结果
压过当前实现文档。

## 9. 治理验收指标

- README 快速开始在 clean environment 可执行。
- 所有公开性能数字链接可复现实验。
- tag、Cargo、镜像、Chart、CHANGELOG 版本一致。
- release workflow 实际产物与注释一致。
- 行为准则和安全报告入口可用。
- 每个关键目录至少两名可审批维护者。
- Stable API 有文档、示例和迁移策略。
