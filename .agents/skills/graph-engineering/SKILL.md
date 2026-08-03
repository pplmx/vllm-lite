---
name: graph-engineering
description: >
  Continuous autonomous engineering loop for the current repository
  (OBSERVE, MODEL, EVALUATE, SELECT, EXECUTE, VERIFY, LEARN, REPEAT),
  backed by a typed knowledge graph (RIL) as cross-session memory.
  Use when the user wants the agent to keep iterating on the repo by
  itself, asks for autonomous/continuous engineering mode, "graph
  engineering", "loop engineering", "keep improving the repository", or
  wants issues tracked as typed nodes and edges with weighted priority
  scoring. Covers graph schema and lifecycle, cross-session loading,
  concurrency locking, scoring, deep-dive budgets, human-intervention
  boundaries, and stop conditions.
---

# Graph Engineering（长期自主工程循环）

## 运行总览

作为长期运行的 Autonomous Engineering Agent，目标不是完成某个预先定义的任务，而是持续自主推进当前 Git 仓库，使项目在每一轮迭代后都变得更正确、更稳定、更安全、更高性能、更易维护。

默认行为：**OBSERVE → MODEL → EVALUATE → SELECT → EXECUTE → VERIFY → LEARN → REPEAT**。
除非触发"人工介入边界"（见第 9 节），不等待确认，不询问"是否继续"。

## 1. OBSERVE

每轮基于仓库最新状态重新观察：

- 代码/架构/依赖
- git status/diff/log
- Issue/TODO/FIXME
- 测试/CI/构建
- 性能/稳定性/安全性/可观测性
- 文档
- 最近变更
- 已有工程知识（见 MODEL）

不要只找孤立 TODO；理解组件、API、数据流、测试、配置、运行时行为之间的关系。

## 2. MODEL 工程图谱

### 2.1 Schema（绑定到 RIL，而不是自然语言描述）

图谱是类型化的节点+边，不是自由文本笔记。RIL（repository-intelligence-layer）的 schema 由本技能自带的 CLI `.agents/skills/graph-engineering/scripts/ril.py`（下称 `ril.py`）强制校验，`.planning/ril/graph.json` 是唯一事实源；一律通过 `ril.py` 读写，**禁止手改 graph.json、禁止新建平行的知识存储**。完整 schema 与 CLI 清单以 `references/ril-schema.md` 为准；`.planning/ril/README.md` 只描述数据存储。

**节点类型**（每个节点必有 `id`, `type`, `status`, `version`, `created_at`, `updated_at`, `touched_round`；下表为各类型的额外必填字段）：

| 节点类型   | 额外必填字段                                                                                                            | 说明                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| component  | —                                                                                                                       | 模块/服务/文件级实体                                                   |
| issue      | —                                                                                                                       | 已识别问题（bug/风险/债务）                                            |
| hypothesis | `confidence`（0-1）                                                                                                     | 未验证的根因猜测                                                       |
| evidence   | `source`（commit hash / 测试名 / 文件行号），append-only                                                                | 支持或反驳某个 hypothesis 的具体观测（测试结果、日志、profiling 数据） |
| decision   | `rationale`、`alternatives_rejected`，不可变                                                                            | 已做出的选择                                                           |
| change     | `commit` hash                                                                                                           | 实际代码修改                                                           |
| task       | `category`（correctness/security/stability/critical-bug/core-feature/performance/test-quality/maintainability/dx/docs） | 可执行的下一步行动，带 priority_score（见 EVALUATE）                   |

**边类型**（有向，语义明确，禁止用无类型的"关联"边）：

| 边类型              | 允许的端点                                        | 语义                                              |
| ------------------- | ------------------------------------------------- | ------------------------------------------------- |
| depends_on          | task→task, component→component                    | 硬依赖                                            |
| causes              | issue→issue                                       | 根因/症状链接，标注是根因还是症状                 |
| blocks              | task→task                                         | 执行阻塞                                          |
| validates / refutes | evidence→hypothesis                               | 证据支持/反驳假设                                 |
| resolves            | change→issue                                      | change 修复 issue                                 |
| supersedes          | decision→decision                                 | 决策变更历史而不是覆盖                            |
| addresses           | task→issue                                        | task 处理某个 issue                               |
| located_in          | issue→component                                   | 问题所在位置                                      |
| part_of             | component→component                               | 子系统层级                                        |
| implements          | change→task                                       | change 交付 task                                  |
| governs             | decision→component/task                           | decision 约束目标                                 |

一个 hypothesis 在没有任何 validates/refutes 边之前，不得被 EVALUATE 当作 fact 使用。

**常用命令**（`ril.py` 即 `.agents/skills/graph-engineering/scripts/ril.py`）：

```bash
ril.py check                      # 一致性检查（孤立节点/循环/无证据 hypothesis）
ril.py tasks --top 10             # active task 按 priority_score 排序取 top-K
ril.py show --id <id> --hops 2    # 拉取节点及其 1-2 跳邻域
ril.py node add --type task --field category=correctness --field severity=...  # 建节点（id 自动分配 TASK-N）
ril.py node set --id <id> --expect-version <v> --field status=resolved         # 乐观更新，版本不匹配会报错
ril.py edge add --type addresses --from <task> --to <issue>                    # 建边
ril.py lock --id <task> --owner <instance> [--minutes 30]                      # 分布式锁
ril.py unlock --id <task>                                                      # 释放锁
ril.py round | ril.py stale --rounds 10                                        # 生命周期维护
```

### 2.2 生命周期与淘汰

- 每个节点有 status：`active` / `stale` / `resolved` / `superseded` / `abandoned`。
- 每轮 MODEL 阶段用 `ril.py round` 推进轮次；`ril.py stale --rounds 10` 把超过 N 轮（默认 10）未被触碰的 hypothesis/task 标记为 `stale`（不删除，保留审计轨迹），EVALUATE 阶段默认跳过 stale 节点，除非新证据重新激活它。
- decision 永不删除，只能被新 decision 通过 `supersedes` 边替代，保留决策演化历史。
- 图谱本身要定期（例如每 50 次 commit 或每周）跑一次 `ril.py check`（孤立节点、循环 depends_on、无证据 hypothesis、长期未闭环的 blocks 边），发现问题作为一个具体 task 提交处理，而不是无限累积。

### 2.3 跨 session 的加载策略

每次 agent 启动是全新 context，不能靠"重读整个图谱"来恢复状态，成本不可控。规则：

- 启动时用 `ril.py tasks --top K` 加载 `status=active` 的 task（按 priority_score 排序取 top-K），用 `ril.py show --id <id> --hops 2` 拉取这些 task 直接关联的 component/issue/hypothesis 子图（1-2 跳），以及最近 N 次 decision。
- 不做全图扫描，除非本轮任务明确是"图谱一致性检查"或"深度探索"（见第 8 节）。
- 如果某个 task 需要更大范围的上下文，允许按需扩展加载（跟着边走），但要在 LEARN 阶段记录"本轮实际使用的子图范围"，供后续 session 参考典型的加载半径。

### 2.4 并发语义

若存在多个 agent instance（Loop Engineering 架构下这是常态）：

- 写入图谱前，对目标节点/边执行乐观锁：`ril.py node set` 必须带 `--expect-version <当前 version>`；版本冲突时 CLI 报错并把节点输出到 stderr，此时重新读取并 diff 合并，而不是覆盖。
- 两个 instance 不得同时对同一 component 下的代码发起 EXECUTE；开始 EXECUTE 前，用 RIL 分布式锁占用对应 task 节点：`python3 .agents/skills/graph-engineering/scripts/ril.py lock --id TASK-x --owner <instance_id>`（默认 30 分钟超时，过期自动释放），结束时 `python3 .agents/skills/graph-engineering/scripts/ril.py unlock --id TASK-x`。**不要**手写 `status=in_progress` 或 `owner=` 字段——RIL schema 没有这些字段，`ril.py` 会直接拒绝。
- evidence 节点只增不改，天然无冲突，鼓励优先通过增加 evidence 而不是编辑已有节点来记录新发现。

## 3. EVALUATE

用加权评分而非严格字典序判断优先级，每个 task 计算：

```text
priority_score = category_weight × severity × confidence × (1 / sqrt(effort)) × unlock_factor
```

- category_weight：正确性/安全性=10，稳定性/关键 bug=8，核心功能=6，性能=5，测试质量=4，可维护性=3，DX=2，文档=1（默认值，可按仓库调整）
- severity：影响范围 × 触发概率
- confidence：该 task 关联的根因判断有多少 validates 证据支撑，未经验证的 hypothesis 打折
- effort：预估实现成本，用于避免"为了刷分做琐碎高权重类别的事"
- unlock_factor：完成后解锁的下游 task 数量/价值，鼓励优先做能解锁后续工作的事

只有当新 task 的 priority_score 显著高于（默认 1.5x）当前正在做的 task 时才切换方向，避免频繁跳变；切换必须在 decision 节点记录原因。

## 4. SELECT

选 priority_score 最高且 `status=active` 的 task，一次聚焦一个主线。允许根据新证据切换，但受上面的切换阈值约束。

## 5. EXECUTE

正常仓库内工程操作（改代码、修 bug、加测试、重构、性能优化、错误处理、可观测性、依赖更新、配置、CI、文档、删除废弃代码）默认自主执行，只要限定在当前仓库且可通过 Git 回滚。

开始前：在对应 task 节点加锁（见第 2.4 节）。

## 6. VERIFY

运行测试/lint/formatter/类型检查/构建/benchmark/静态分析。失败时：分析根因 → 修根因 → 重新验证。

硬性禁止：删测试、跳测试、降低断言/阈值、注释失败用例、修改质量标准来制造"通过"。无法可靠修复时回滚本轮改动，并在图谱中把对应 hypothesis（如果修复基于某个根因假设）标记为 `refuted`，附上 evidence。

## 7. LEARN

按第 2.1 节的 schema 写入节点和边，而不是自由文本日志。区分 Fact/Hypothesis/Evidence/Decision 必须体现在节点类型上，不是靠文字语气区分。

Commit 时在 message 里引用相关 task/issue 节点 id，保证代码历史和图谱可以互相追溯。

## 8. 深度探索（无明显 TODO 时）

主动做 Repository Intelligence Deep Dive，寻找隐藏 bug、边界问题、并发问题、错误处理缺陷、资源泄漏、性能瓶颈、测试缺口、安全风险、架构耦合、技术债务，优先形成"证据 → 根因 → 修复 → 验证"闭环。

硬性预算约束：

- 单轮深度探索最多产出 3 个新 task 节点，否则说明范围没收敛，需要先合并/归类。
- 单次 commit 的 diff 不超过某个阈值（默认 300 行，特殊重构除外并需在 decision 中说明理由）。
- 若连续 2 轮深度探索新增 task 的 priority_score 均低于当前阈值（默认 3.0），停止深度探索，转入停止条件评估。

## 9. 人工介入边界

只有以下情况暂停等待人工确认：

- push 到 main/master 或强制 push
- 删除远程分支
- 正式发布版本或包
- 不可逆的生产环境操作
- 不可逆的数据删除或破坏性数据库迁移
- 需要访问无权限的秘密/凭据/敏感数据
- 明显超出当前仓库权限范围
- 无法合理回滚且可能造成重大外部影响的操作

## 10. 停止条件

满足全部以下条件才停止：

1. 图谱中不存在 `status=active` 且 priority_score 高于阈值（默认 3.0）的 task。
2. 所有 severity 高的 issue 节点，status 为 `resolved` 或有明确 decision 记录暂缓原因。
3. 最近一次 VERIFY 全绿。
4. 连续 2 轮深度探索无法产出高于阈值的新 task（见第 8 节）。
5. 图谱一致性检查（第 2.2 节）无未处理的孤立/循环节点超过 N 个。

否则继续 REPEAT，直到用户主动中止或以上全部满足。
