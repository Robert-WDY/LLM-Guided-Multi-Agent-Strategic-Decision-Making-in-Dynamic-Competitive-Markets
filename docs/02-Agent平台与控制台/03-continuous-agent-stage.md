# Agent 连续决策与 4-Agent 闭环阶段

## 当前完成状态

本阶段不增加市场动作，不实现通信、公共品、社会福利或不完全信息。工作集中在连续决策、历史利用、跨轮计划、统一审计和四 Agent 稳定编排。

| 能力 | 状态 |
| --- | --- |
| 当前公司、市场、竞争者、风险、事故和动作约束 | 完成 |
| 最近3轮、最近5轮趋势和重大事件记忆 | 完成 |
| 带期限、子目标和触发器的 Current Plan | 完成 |
| `full` 与 `state_only` 上下文消融 | 完成 |
| 确定性 ResultAnalyzer | 完成 |
| 四 Agent 并发决策、动作锁和单次统一结算 | 完成 |
| 完整 RoundEvent 与 Replay | 完成 |
| 状态响应固定案例矩阵 | 已实现正常、产能压力、风险预警、主动事故、低现金五类状态 |
| 4 个真实 LLM × 20轮 | 尚未直接运行，按分级路径推进 |

## Current Plan v1.1

计划包含：

- `plan_id / created_round / horizon / expires_round`；
- `objective / pending_subgoals / priorities`；
- `replan_triggers / handled_trigger_event_ids`；
- 动态现金、价格和投入约束；
- `replanned / replan_reason` 审计信息。

计划只在初次建立、到期、经营阶段变化、新风险信号、公司事故、产能或缺货阈值、重大市场变化、预测或目标失败时重建。无触发器时保持原 `plan_id`。

## 上下文消融

`DecisionContext v1.4` 支持：

- `full`：当前状态 + 最近历史 + 趋势 + 关键事件 + Current Plan；
- `state_only`：仅当前状态和当前动作约束，隐藏跨轮历史、关键事件和计划。

真实 Doubao 小样本使用相同 Balanced Persona、市场 Seed 43，各运行5轮：

| 指标 | Full Context | State Only |
| --- | ---: | ---: |
| 累计利润 | 10,896,471 | 12,088,249 |
| 最终份额 | 261,935 | 249,955 |
| 最终声誉 | 596,229 | 582,811 |
| 最终韧性 | 210,000 | 104,353 |
| 主动投入 | 10,450,000 | 7,900,000 |
| 终局企业价值 | 48,093,630 | 48,584,947 |
| 连续重复动作率 | 25% | 75% |

该样本说明完整上下文显著减少机械重复，并促使 Agent 根据风险和历史调整韧性、价格与投入；但它也增加了投入，本样本的利润和终局价值略低。只有一个 Seed，不能据此断言历史上下文普遍提高收益。

## RoundEvent v1.2

每轮保存：

- 完整 `state_before / state_after`；
- 完整 Observation 和 DecisionContext；
- Planner Output、原始模型输出、请求动作、最终动作和联合动作；
- Validation 调整、错误、模型修复重试次数和 fallback；
- Random Draw Summary、Step Result 和确定性 ResultAnalysis；
- Token、Latency，以及暂不可计算时为 `null` 的成本字段。

旧版 v1.0/v1.1 日志仍可读取；新增字段对旧日志使用空值默认。

## 4-Agent × 20轮验收

Mock 验收结果：

- 4 Agents、20轮、80次决策全部完成；
- 所有 Agent 使用同一 State Version 和 State Hash；
- 每轮只有一次 MarketEnv 结算；
- Duplicate Action 为0；
- Partial State Update 为0；
- Fallback 为0；
- RoundEvent 完整率100%；
- Replay Match 100%。

结果位于 `runs/four-agent-20-round-mock-v1/`。

## 1000 Episode 稳定性验收

5种市场 × 每种200 Seed × 20轮，共1000 Episode、20,000轮：

- Episode 完成 1000/1000；
- State Invariant Failure 0；
- 负单位贡献执行 0；
- 现金储备违规 0；
- 恢复阶段继续降价 0。

结果位于 `runs/acceptance/continuous-agent-stage-v1.json`。

## 下一步顺序

1. 用多个 Seed 重复 `full vs state_only`，报告重复动作率、终局价值和 Bootstrap 区间；
2. 用真实模型和多个 Seed 批量运行已实现的五类固定状态响应矩阵；
3. 依次运行 `1 LLM + 3 Rule`、`2 LLM + 2 Rule`；
4. 最后运行 `4 LLM × 20轮`；
5. 完成这些验收后再引入合作、公共品和通信。
