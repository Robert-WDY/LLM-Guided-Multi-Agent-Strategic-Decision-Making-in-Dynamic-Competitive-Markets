# Stage 6.5：真实 LLM 战略建议采纳实验

日期：2026-08-24。

## 1. 目标与停止规则

本阶段要验证：`Persona + Belief + Pareto Planner` 是否能稳定改变真实 LLM 的战略行为，并改善建议后三轮的企业价值与有限长期 Regret。

原始完整矩阵为 10 Seeds × 3 Persona × 5 条件 × 10 轮。如果每轮都调用真实模型，需要 1,500 次付费调用。为节约 Token，本阶段预注册为渐进实验：

1. 每个 10 轮 Episode 只在第 7 轮调用一次真实模型；
2. 第 1–6 轮使用与条件无关的固定经济基线，积累公开历史；
3. 第 8–10 轮继续使用相同固定基线，观察真实动作的三轮后果；
4. 先跑 5 次真实 Smoke；
5. Smoke 工程通过后扩到 3 Seeds × 3 Persona；
6. 只有所有 Persona 的平均和最坏门禁通过才扩到 10 Seeds；失败人格只允许少量补充 Seed 判断负尾部是否偶然。

因此完整 10-Seed 矩阵不是必须消费的预算，而是研究门禁通过后的扩展上限。

## 2. 五个条件

所有真实调用使用同一个豆包模型、相同 Seed、相同 Persona、相同第 7 轮经济状态，只改变可见战略信息：

| 条件 | Belief | Opponent/Utility | Advisor |
|---|---|---|---|
| Persona Only | 关闭 | 关闭 | 关闭 |
| Action Belief | 价格方向 Belief | 关闭 | 关闭 |
| Strategy Utility | 价格 Belief | 公开策略分布和效用推断 | 关闭 |
| Bayesian v2 | 开启 | 开启 | 旧 Bayesian Advisor |
| Pareto v4 | 开启 | 开启 | Stage 6.4 Pareto Planner |

这种设计同时测量弱 Belief、策略/效用 Belief、旧 Advisor 和 Pareto Advisor 的增量效果。

## 3. Advisor Adoption Trace

`AgentRoundTrace` 新增带独立 Hash 的 `AdvisorAdoptionTrace`，记录：

- Advisor 推荐候选和完整经济动作；
- LLM 请求动作；
- Controller 合法化后的最终动作；
- 完全采纳、目标方向采纳、部分采纳或拒绝；
- 完整动作对齐率和目标维度对齐率；
- Advisor 排名、LLM 实际选择的候选及排名；
- LLM 自己输出的策略理由；
- 可验证的 Trace Hash。

“模型请求”与“最终动作”分别保存，避免把 Controller 的安全调整误判成 LLM 拒绝。对于 Pareto 推荐，允许 LLM 不完全复制预算，只要沿推荐目标方向调整就记为目标采纳；选择候选菜单中的其他动作则记录候选排名，不伪装成接受第一建议。

RoundEvent 升级为 `agent-round-event-v1.10.0`，旧事件版本继续可读。

## 4. 工程缺陷与修复

首次 Mock Smoke 发现，第 7 轮前五组状态不一致。原因有两个：

1. 原 Mock Policy 会读取不同条件下的 Planner 约束，导致非付费轮已经受 Belief/Advisor 条件影响；
2. Rule 对手的 seeded variation 使用 Episode ID，若条件名进入 Episode ID，即使市场 Seed 相同，对手动作也会不同。

修复方法：非付费轮改成完全忽略 Persona/Belief/Advisor 的固定经济基线；同一 Seed + Persona 的所有条件复用同一 Episode ID。修复后的 Smoke 中，五组第 7 轮前 `pre_decision_economic_state_hash` 完全一致。

新增 `SelectiveDecisionModelClient` 和 `RealModelCostGuard` 集成。每个 Episode 只有预注册轮允许访问真实 Provider，调用前按 32K 输入、4K 输出预留预算，Provider 返回后记录实际 usage。未授权、调用数超限或 Token 超限都会中止。

## 5. 真实 Smoke

真实 Smoke 使用 `doubao-seed-2-0-lite-260215`，1 Seed × 极端激进 × 5 条件：

- Calls：5；
- Input Tokens：90,410；
- Output Tokens：2,668；
- 保守成本：0.101082 CNY；
- 五组付费决策前状态完全一致；
- 所有结构化输出、动作结算、Trace 和 usage 完整。

弱 Belief 与策略/效用 Belief没有改变该样本动作。旧 Advisor 和 Pareto 都改变了动作并提高终局价值。Pareto 推荐 `maintain` 运营候选，LLM 没有照搬第一名动作，而选择了候选排名第二的 `price_cut_large`，保留相同广告与服务预算；Trace 正确记录为 `partial`，而不是错误地记为采纳。

## 6. 渐进真实结果

3-Seed Pilot 共 45 次真实调用。极端激进通过方向门禁；短期逐利与风险防御出现负的最坏配对，因此没有直接扩展到 10 Seed。只为两个失败人格增加 Seed 4–5，共 20 次调用，验证负尾部是否重复。

最终自适应样本为：

- 极端激进：Seeds 1–3；
- 短期逐利：Seeds 1–5；
- 风险防御：Seeds 1–5；
- 每个 Seed/Persona 单元五个真实条件；
- 共 13 个配对单元、65 次真实调用。

### 6.1 Pareto v4 相对 Strategy Utility 无 Advisor

金额单位为分，Regret 是五个真实 LLM 动作在建议后三轮终局价值中的有限反事实 Regret。

| Persona | 配对 | 动作改变 | 推荐采纳 | 平均企业价值差 | 最坏差 | 正/负 | 平均 Regret 差 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 极端激进 | 3 | 3/3 | 1/3 | +10,581,714 | +6,624,987 | 3/0 | -10,581,714 | 通过 |
| 短期逐利 | 5 | 5/5 | 5/5 | +33,371 | -1,919,523 | 3/2 | -33,371 | 失败 |
| 风险防御 | 5 | 5/5 | 5/5 | +247,875 | -638,557 | 2/3 | -247,875 | 失败 |

总体 13 个配对中，Pareto 使动作 13/13 发生变化，推荐采纳 11/13（84.62%）；平均企业价值增加 2,550,106、Regret 降低 2,550,106，但只有 8 组提高、5 组下降，最坏下降 1,919,523。总体均值主要被激进人格的大幅收益拉高，不能掩盖另外两种人格的负尾部。

### 6.2 旧 Advisor

旧 Bayesian v2 的采纳率为 12/13（92.31%），高于 Pareto，但结果更差：

- 极端激进 3/3 提高；
- 短期逐利仅 1/5 提高，平均 -821,678；
- 风险防御 0/5 提高，平均 -1,041,209；
- 总体为 4 组提高、9 组下降。

这直接证明“采纳率高”不等于“战略模块有效”。

### 6.3 Belief

Action Belief 相对 Persona Only：13 组中 7 组动作不变、4 组价值提高、2 组下降，平均 +187,144，最坏 -686,881。短期逐利 5/5 完全不变；风险防御虽然平均略增，但存在两个负配对。

Strategy Utility 相对 Action Belief：7 组动作不变、3 组提高、3 组下降，平均 -544,759，最坏 -7,858,357。该大幅负值来自极端激进的一条路径；短期逐利仍完全不变。

当前 Belief/对手效用推断能改变部分真实 LLM 决策，但没有形成跨 Persona 的稳定下游收益。现阶段不应继续增加欺骗、隐藏类型或合作 Belief。

## 7. Token 与费用

最终有效实验：

- Model：`doubao-seed-2-0-lite-260215`；
- Calls：65；
- Input Tokens：1,155,180；
- Output Tokens：34,175；
- Total Tokens：1,189,355；
- 保守预算记账：1.291880 CNY；
- 按项目此前固定的 0.0006 元/千输入 Token、0.0036 元/千输出 Token 快照换算约 0.816138 CNY，最终账单以 Provider 为准。

正式合并结果：`runs/stage6.5-real-adoption/summary.json`，SHA-256 为 `208292A46353129E53809F76626A7903E566E6B978EC03A1089D64A90D7B1768`。

## 8. 结论与下一步

工程结论成立：Pareto Advice 能进入真实 LLM 上下文，真实 LLM 会采纳、部分采纳或拒绝，行为和理由均可记录与回放；`engineering_passed=true`。

研究结论未通过：Pareto 能稳定改变动作，但不能在短期逐利和风险防御人格上同时保护最坏企业价值；`directional_gate_passed=false`、`stage65_complete=false`。因此按成本门禁停止，没有运行 Seeds 6–10，也没有运行 Known/Mixed/Holdout 对手泛化。

下一步不是增加更多真实样本掩盖失败，而是离线分析五个负配对：比较推荐候选、LLM 最终候选、Persona 效用底线和真实三轮损失，修复“建议被高比例采纳却仍产生尾部损失”的 Planner→LLM 接口。修复必须先通过零模型反事实和现有 65 条冻结输出 Replay，再申请少量失败 Seed 复测。
