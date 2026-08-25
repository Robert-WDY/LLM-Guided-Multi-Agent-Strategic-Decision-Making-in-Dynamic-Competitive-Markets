# P0–P2 不可变 Agent 真实落地验收

日期：2026-08-25。

## 验收问题

本阶段不再只问“版本合同是否能通过单元测试”，而是验证以下完整链路：

```text
当前源码、依赖、Prompt、Persona、战略栈
→ 注册不可变 Agent Version
→ Registry 解析
→ 构造 AgentRuntime
→ immutable_v1 Episode
→ Observation / Belief / Advisor
→ Agent Decision
→ Controller Settlement
→ Round Trace
→ 五层 Replay
```

成功标准是每个 Runtime 确实由权威 Version 构造，逐轮 Trace 归属完整，并且经济、信息、交互、博弈模块和 Agent Version 归属可以同时重建。

## 新增运行能力

`CurrentAgentVersionBuilder` 将当前行为栈注册为内容寻址版本。行为输入包括确定性源码 ZIP、直接依赖锁、Prompt/Mock Policy 源码、Persona Profile Hash、公开信息策略栈、Candidate Generator、Advisor Adoption、Resolver 与安全版本。

`RegisteredAgentRuntimeFactory` 不接受临时拼装配置。它先验证：

- Registry Manifest 与 Artifact 完整；
- 当前源码 ZIP 与登记 Hash 一致；
- 当前依赖锁与登记 Hash 一致；
- Market Environment 和 Config Hash 兼容；
- Persona Catalog 和 Profile Hash 一致；
- 同一 Episode 的 Controller Mode 一致。

验证后才构造 Provider Client、Context Builder、Memory 和 Instance Binding。源码或依赖变化后，旧版本仍能从 Registry 读取，但不能以当前新代码冒充旧版本执行；要执行旧版本需要未来的归档运行环境或容器能力。

## 严格回放修复

真实落地审计发现 Round Event 已升级到 v1.10/v1.11，而 Information、Belief、Interaction Replay 的严格版本集合停留在 v1.9。新事件因此没有启用全部缺字段门禁。现已把 v1.10、v1.11 和当前 v1.12 纳入严格回放集合。

另外补充远程 Provider 审计元数据：Request ID、请求和响应时间、响应模型、服务端创建时间与 system fingerprint。Tier C 仍不能保证未来重新调用得到相同文本，但可以证明记录中的原始输出来自哪次 Provider 响应，并确定性回放输出之后的处理。

## 确定性完整 Episode

配置：

- Seed：20260825；
- 4 Agents × 5 Rounds；
- 四种不同 Persona；
- Public Information；
- Belief、Opponent Model、Utility Inference、Pareto Reliable v5 开启；
- Communication、Cooperation 关闭；
- Provider：第一方确定性 Mock。

结果：

| 指标 | 结果 |
|---|---:|
| 唯一 Agent Version | 4 |
| 完整 Trace | 20/20 |
| Version Attribution | 100% |
| Economic Transition | 5 |
| Economic Replayed State | 6 |
| Information Snapshot | 20 |
| Interaction Closure | 5 |
| Game Theory Binding | 60 |
| Hidden State Leak | 0 |
| 真实模型调用 | 0 |

最终 State Hash：`sha256:a61617796c8672bc2245ba912107060ad8e09515fd43f7282df56e169bb1fed4`。

内部 Report Hash：`sha256:7e11ebdf66024a5d6e5d0282eb3dd212a72310fd93b8787c9f55b10bd8df51f7`。

相同 Seed 重新运行后，四个 Version ID、Roster Hash、最终 State Hash 和 Report Hash 完全一致。当前四个版本已晋升为 `validated`。

## 最小真实模型 Smoke

在确定性门禁通过后，最终规范 Smoke 只允许一次 DeepSeek 调用：

- 1 DeepSeek + 3 Mock；
- 只推进第 1 轮；
- `max_schema_attempts=1`；
- 最多 32K 输入、4K 输出；
- 高峰价格保守预算上限 0.08 元。

最终结果：

| 指标 | 结果 |
|---|---:|
| 模型 | deepseek-v4-flash |
| 成功调用 | 1 |
| 输入 Token | 14,742 |
| 输出 Token | 468 |
| 高峰保守费用估计 | 0.031356 元 |
| 决策状态 | submitted |
| Version 归属 | 通过 |
| Economic Replay | 通过 |
| Information Replay | 通过 |
| Interaction Replay | 通过 |
| Game Theory Replay | 通过 |
| Agent Attribution Replay | 通过 |
| Hidden State Leak | 0 |
| Provider Request ID | 已记录 |
| Provider Fingerprint | 已记录 |

远程 Agent Version：`sha256:1d62336026d2b95b954ad79acc6e8f29f9cbaa6efc7df32c36e3ef6e951f8ea3`。

内部 Report Hash：`sha256:c34a406533dfe95fd709e9ecd2384308e43c162bbc0d94fb35732bd96944909a`。

该结果只证明注册版本可以构造真实 Provider Runtime，并形成完整 Trace 与 Replay。单轮成功不能证明战略收益改善，因此远程版本只晋升为 `candidate`。

为透明记录本阶段实际费用：在补齐 Provider Audit 合同之前曾运行过一次诊断调用，使用 14,742 输入 Token、469 输出 Token；它暴露了 Request ID、响应模型和 fingerprint 未进入 Round Trace 的审计缺口，不作为最终验收样本。修复后上述最终规范调用使用 14,742 输入 Token、468 输出 Token。因此本阶段共发生 2 次付费调用，合计 29,484 输入 Token、937 输出 Token；按同一高峰双倍费率保守估算总费用为 0.062716 元。

确定性验收汇总文件 SHA-256：`83D17E4E3A465FFF1285849A4BD851933569A124438BA1C77019F7D3C04011ED`。

最终真实 Smoke 汇总文件 SHA-256：`93EBD3B098891ED7076B17AFA6D84E7FC4054962846DC424AB80E1595C6ACF30`。

## 实施中发现的问题

1. 首次验收把 Replay 返回的“初始状态 + 结算状态”误当成 Transition 数，导致正确的五轮运行被判失败。修复为分别记录 5 条 Transition 和 6 个 State。
2. 第二次运行复用了 JSONL 并追加事件，导致 Interaction Replay 看到重复轮次。修复为每次运行先清理该实验唯一明确的事件文件，Registry 和历史版本不删除。
3. 新事件没有进入 Replay 严格版本集合。修复后 v1.12 强制完整 Snapshot、通信身份和历史绑定。
4. 首次真实调用缺少 Provider Audit。补充合同后再次以相同单次预算运行，确认真实服务返回 Request ID、响应模型和 fingerprint。

这些问题均属于验收工具或审计覆盖缺口，没有修改 Market 经济公式，也没有掩盖失败结果。

## 结论边界

P0–P2 已从合同测试进入实际运行：当前系统可以注册、解析、构造、执行和回放不可变 Agent Version。仍未完成的是通过归档源码自动启动旧运行环境，以及 P3 Population Archive、P4 Matchmaker、P5 Mutation/Training/Promotion Gate。
