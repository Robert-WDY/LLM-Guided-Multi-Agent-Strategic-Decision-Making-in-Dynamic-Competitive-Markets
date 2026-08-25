# Stage 6.4：约束式 Pareto 决策可信化

日期：2026-08-24。

## 1. 阶段目标

Stage 6.3 证明固定标量权重无法同时保护企业价值、竞争位置、尾部风险和 Regret，并且存在“一步更优、长期更差”的路径。Stage 6.4 不再细调一个总分，而是把长期战略建议改造成两段式决策：

1. 先用真实市场 Rollout 删除违反绝对价值、竞争差距、最坏情景和人格效用底线的候选；
2. 再从安全候选的 Pareto 前沿中，根据当前名次、竞争差距和剩余轮数选择动作。

阶段完成标准不是平均收益变好，而是三个目标人格都必须在开发集选出同一已声明方案，并在隔离保留集同时通过企业价值、最坏配对、竞争差距、第一名率、长期 Regret 和实际动作变化闸门。

## 2. 决策器实现

新增 `ParetoPlannerSpec`、`ParetoSituation`、`ParetoCandidateAssessment` 和 `ParetoPlannerDecision`。所有合同使用严格字段验证和独立 Decision Hash，候选评估记录：

- 相对 Rule 运营基线的预期企业价值约束；
- 相对最强对手竞争差距约束；
- 最坏风险调整价值约束；
- Persona 确定性等价值约束；
- 是否被其他安全候选四维支配；
- 是否位于安全 Pareto 前沿。

情境分为：

- `protect_lead`：已经领先，优先保护人格长期价值；
- `catch_up_early`：尚有较多轮次，按配置优先经营价值或竞争位置；
- `catch_up_late`：接近终局且落后，可在预先声明的预算内增加竞争优先级。

任何时候 Rule 运营基线都保留为安全回退。正式选中的 `strict_value` 不允许预期价值或最坏情景相对基线下降；只有满足全部底线的候选才有资格进入 Pareto 前沿。

## 3. 因果对照缺陷与修复

首轮正式运行虽然平均结果明显提高，但阶段没有通过。调查发现两个实验口径问题：

1. Rule 对照直接从真实 `MarketState` 生成动作，处理组却从公开 Observation 重建预测状态，两组的信息权限不一致；
2. Regret 使用从真实状态重新生成的候选，而不是 Agent 当轮实际获得的公开候选，评价集合不一致。

修复后，所有条件都从同一 company-scoped 公开 Observation 重建相同 Forecast，并共享相同预测随机流。Rule 对照使用该 Forecast 中的 `maintain` 运营基线；处理组只在同一候选集合上增加战略选择。Regret 也只在当轮真实可选的候选集合内计算。

第二轮运行显示极端激进人格的终局企业价值和竞争指标提高，但单轮 Regret 增加。由于 Stage 6 的原始问题正是“局部 Best Response 不等于长期战略最优”，用单轮 Regret 否决长期投资与阶段目标冲突。因此最终同时保留：

- 单轮事后 Regret：用于诊断即时机会成本；
- 两轮真实反事实 Regret：对每个实际公开候选执行固定对手策略和真实 `MarketEnv`，作为正式战略可靠性闸门。

这不是取消 Regret 约束，而是把它从即时指标校正为长期指标。

## 4. 正式实验矩阵

- Seeds：1–10，共 10 个共同 Seed；
- Persona：极端激进、短期逐利、风险防御；
- 对手池：Known、Mixed、Holdout；
- 每局：5 轮；
- 在线预测：2 轮 × 2 情景；
- 条件：Rule、Persona、25% 竞争目标、5 个约束式 Pareto 规格；
- 总计：720 个 Episode、3,600 个真实市场回合；
- 开发集：Seed 1–5 × Known/Mixed，每人格 10 个配对；
- 保留集：Seed 6–10 的全部对手池，加 Seed 1–5 的 Holdout，每人格 20 个配对。

开发集只按预先声明顺序选择第一个通过全部门禁的 Pareto 规格，随后在保留集只验收该规格。三个 Persona 均选择 `pareto_strict_value`。

## 5. 保留集结果

下表均相对公开信息 Rule 对照，金额单位为分。

| Persona | 平均企业价值差 | 最坏企业价值差 | 平均竞争差距差 | 第一名率变化 | 两轮 Regret 差 | 单轮 Regret 差 | 非基线动作率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 极端激进 | +2,185,481 | 0 | +1,366,540 | +15 个百分点 | -368,692 | -79,516 | 37% |
| 短期逐利 | +2,258,065 | 0 | +1,336,334 | +15 个百分点 | -594,552 | -468,277 | 35% |
| 风险防御 | +2,250,324 | 0 | +1,349,298 | +15 个百分点 | -388,125 | -255,321 | 39% |

三个 Persona 的 60 个保留集配对中，没有任何企业价值负配对；最坏差为 0，代表最差情况退回 Rule，而不是强行采取建议。所有 Persona 的平均企业价值、竞争差距和第一名率提高，长期 Regret 下降，并且动作变化率明显高于 5% 的非平凡门槛。

正式运行中 `protect_lead` 出现 438 次，`catch_up_late` 出现 12 次；当前五轮市场样本多数从领先状态出发，因此不能据此宣称早期追赶策略已获得充分覆盖。

## 6. 在线晋升

保留原 `public_rollout_v3` 语义不变，新增独立 `advisor_mode=pareto_rollout_v4`：

- 仍只允许 `information_mode=public`、启用 Belief 和公开 Opponent Model；
- 使用与 v3 相同的公开 Forecast 和真实市场有限视野 Rollout；
- 使用已通过开发集/保留集的 `strict_value` 约束策略选择安全 Pareto 候选；
- 输出完整 `pareto_decision`、`selection_situation`、Decision Hash 和正式结果 SHA-256；
- 建议保持非绑定，Agent 可以根据现金与硬约束拒绝；
- API、Prompt、Manifest 和 Advisor Replay 已支持 v4；篡改 Pareto 推荐或证据字段会验证失败。

晋升证据 SHA-256 为 `66540760D0C250FADD2DF023124BAC7CACE83116379B0D6D9426CDA0239228EE`。

## 7. 验收与成本

- 正式实验工程检查全部通过；
- `strategic_reliability_stage_complete=true`；
- 正式 JSON 重复运行字节级一致；
- 212 项全量测试通过；
- 本阶段相关文件 Ruff 通过；
- `Calls=0`；
- `Prompt Tokens=0`；
- `Completion Tokens=0`；
- `Estimated Cost=0 CNY`。

结果文件：`runs/stage6.4-pareto/summary.json`。重复结果：`runs/stage6.4-pareto/summary.repeat.json`。

## 8. 结论边界

“战略决策可信化阶段完成”在本项目中表示：合同、公开信息边界、长期反事实决策、在线接入、Replay、10-Seed 开发/保留集门禁已经完成。它不表示：

- 已在真实 LLM 上证明采纳率或收益提升；
- 已获得统计显著性或现实市场外部有效性；
- 已解决 Nash 均衡、PSRO 或机制设计；
- 已充分覆盖早期落后追赶、更多 Persona 和更多市场参数。

真实 LLM 仍应保持成本门禁。下一步只需做小额、预注册的 v3/v4 采纳与拒绝 Smoke，而不是重新开发战略核心。
