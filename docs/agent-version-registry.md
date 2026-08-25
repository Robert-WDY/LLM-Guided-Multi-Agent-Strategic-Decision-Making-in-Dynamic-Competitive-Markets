# 不可变 Agent 版本注册表

## 阶段定位

该模块解决“历史实验究竟由哪一套 Agent 策略产生”的可归因问题。`agent_id` 仍用于运行时称呼，真正的策略身份改为 `AgentVersionID = SHA256(CanonicalJSON(AgentBehaviorSpec))`。

当前实现完成 P0–P2 核心能力及真实运行落地验证；Population Archive、Matchmaker、训练和晋升评估属于 P3–P5，尚未伪装成已完成能力。

## 五层身份

- `family_id`：组织一个长期策略家族，不参与运行。
- `AgentVersionManifest`：不可变策略定义，任何行为字段变化都会产生新 ID。
- `AgentInstanceBinding`：一个版本在某个 Episode、Company 席位上的实例。
- `CheckpointRef`：跨 Episode 学习状态；进入正式实验后作为内容寻址 Artifact，并生成新版本。
- `AgentLifecycleRecord`：展示名、标签、评级、比赛数和生命周期；可更新但不参与版本 ID。

## 行为身份覆盖范围

`AgentBehaviorSpec` 固定以下行为输入：

- Provider、模型修订、采样参数、请求与响应 Schema；
- Persona Profile Hash 和 Persona Planner；
- System、Decision、Communication、Parser Prompt 的内容 Hash；
- Observation、Visibility、Decision Context、Memory、Critical Event Selector；
- Belief、Opponent Model、Utility Inference、Advisor、Repeated Game、Cooperation；
- Candidate Generator、Advice Adoption、Action Parser、Resolver、安全和 Fallback；
- 源码归档、依赖锁、代码提交和容器摘要；
- 可兼容的 Market、Observation、Event 和信息模式；
- 可选 Checkpoint。

展示名、标签、Rating、比赛数和 Champion 状态位于独立生命周期表，不改变 `AgentVersionID`。

## 存储与不可变规则

`AgentRegistry` 使用文件内容寻址对象库和 SQLite 索引：

```text
.agent-registry/
├── registry.sqlite3
├── objects/sha256/<prefix>/<digest>
└── versions/sha256/<prefix>/<digest>/manifest.json
```

Manifest 只能插入。SQLite Trigger 禁止修改或删除 `agent_versions`；相同 Behavior Spec 重复注册返回原版本；不同内容只能创建新版本。读取时同时验证：

- Manifest 是规范 JSON 字节；
- Manifest Hash、Behavior Hash 和 Version ID 一致；
- 所有 Prompt、源码、依赖与 Checkpoint Artifact 存在；
- Artifact 长度和 SHA-256 一致；
- Lineage 父版本已经注册。

## Episode 接入

`CreateEpisodeRequest` 新增：

```json
{
  "agent_versioning_mode": "immutable_v1",
  "agent_version_ids": {
    "company_A": "sha256:...",
    "company_B": "sha256:..."
  }
}
```

`immutable_v1` 必须为每家公司提供一个已注册版本。Controller 在创建 Episode 前解析全部版本，无法解析、缺席或多余的公司都会返回 422。`EpisodeManifest` 保存完整 `agent_roster` 和 `agent_roster_hash`。

`CurrentAgentVersionBuilder` 会把当前源码、配置、依赖、Prompt/Policy 和 Persona 注册为正式版本。源码归档使用固定顺序、固定时间戳和无压缩的确定性 ZIP；依赖锁保存声明依赖和实际直接依赖版本。

`RegisteredAgentRuntimeFactory` 从 Registry 解析版本，重新校验 Market Config、Persona Hash、源码归档和依赖锁，然后构造 ModelClient、DecisionContextBuilder、EpisodeMemory 与不可变 Instance Binding。当前 Checkout 与登记源码不一致时拒绝执行，不会让旧 ID 静默运行新代码。

`AgentRuntime` 接收不可变 `AgentInstanceBinding`，并拒绝公司或 Agent 身份不一致的绑定。Coordinator 在 `immutable_v1` Episode 中采用 fail-closed：运行时绑定必须与 Controller Roster 完全相同，否则在获取 Observation 和生成决策前停止。

`AgentRoundTrace` 和通信生成 Trace 冗余保存 Family、Version、Instance、Manifest、Behavior、Prompt Bundle、Source Bundle 与 Checkpoint Hash。Round Event 当前为 v1.12；真实 Provider Trace 还保存请求时间、响应时间、Request ID、响应模型、Provider 创建时间和 system fingerprint。`replay_agent_attribution` 会重新解析 Registry，并逐公司、逐回合验证 Trace 归属；替换任一版本字段都会失败。

旧入口继续明确标记为 `legacy`，用于兼容历史实验。它们不会被错误描述为已完成不可变版本归因。新的正式 Self-play 和版本比较实验必须使用 `immutable_v1`。

## 复现等级

- `A_deterministic`：本地或确定性 Rule/Mock，权重、依赖、代码和 Prompt 均可固定。
- `B_provider_seeded`：Provider 提供可固定快照或请求 Seed。
- `C_recorded_output`：只能确定性重放已记录输出之后的 Parser、Resolver 和 Market；不能保证重新调用远程模型得到同一输出。

## 已完成验收

- 相同 Spec 的 ID 一致；Prompt 改一个字符或 Advisor 版本变化会换 ID；
- 修改 Rating、标签和展示名不换 ID；
- 父子版本并存，旧版本不变；
- Manifest 和 Artifact 篡改均被发现；
- `immutable_v1` 无法解析或不完整 Roster 时拒绝创建；
- Roster Hash 可重建；Trace 版本替换导致 Agent Attribution Replay 失败；
- 同一 Agent Version 的不同 Episode Instance 不共享 `EpisodeMemory`。

## 真实落地结果

确定性验收使用四个正式 Mock 版本，Persona 分别为 `balanced_v1`、`aggressive_v1_extreme`、`risk_guarded_v1` 和 `profit_myopic`，共同开启 Public Information、Belief、Opponent Model、Utility Inference 和 `pareto_reliable_v5`。Seed 20260825 下完整运行 5 轮：

- 4 个唯一 Agent Version，20/20 Trace 完整归属；
- 5 条 Economic Transition、6 个回放状态；
- 20 个 Information Snapshot、5 个 Interaction Closure；
- 60 个 Game Theory Trace Binding；
- 隐藏状态泄漏 0；
- 相同 Seed 重跑的 Version ID、最终 State Hash 和 Report Hash 完全一致；
- 当前四个版本已从 `draft` 经过 `candidate` 晋升到 `validated`。

随后只运行一次最终规范 DeepSeek Smoke：1 个 `deepseek-v4-flash` + 3 个 Mock，仅推进第 1 轮，禁用 Schema 重试。真实版本为 Tier C；决策成功提交并绑定正确版本，所有回放通过，Request ID、响应模型和 fingerprint 均已记录。最终调用使用 14,742 输入 Token、468 输出 Token；按 2026-08-25 DeepSeek 官方价格并按高峰双倍费率保守估算 0.031356 元。该远程版本只晋升到 `candidate`，未因单轮 Smoke 被错误标为 `validated`。

阶段总费用单独审计：Provider Audit 字段补齐前的一次诊断调用使用 14,742 输入 Token、469 输出 Token，修复后最终规范调用使用 14,742 输入 Token、468 输出 Token。本阶段实际共 2 次付费调用，合计 29,484 输入 Token、937 输出 Token，高峰双倍费率保守估算 0.062716 元。第一次调用仅用于暴露审计缺口，不计入最终规范 Smoke 结果。

正式产物位于：

```text
runs/immutable-version-acceptance/
runs/immutable-real-smoke/
```

## 尚未完成

P3–P5 仍需单独实现和验收：Population/Payoff Archive、公司席位轮换 Matchmaker、Known/Mixed/Holdout 隔离、Mutation/Training/Checkpoint 工作流，以及同时考虑企业价值、竞争差距、尾部风险、Regret 和泛化的晋升门禁。
