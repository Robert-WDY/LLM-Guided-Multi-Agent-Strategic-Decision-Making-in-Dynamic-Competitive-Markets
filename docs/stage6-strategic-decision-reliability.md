# Stage 6：战略决策可信化 v1

日期：2026-08-24。

## 1. 阶段定位

当前系统已经具备 `Opponent Model → Utility Inference → Bayesian Advisor → Agent Decision` 工程链路，但 Stage 5.1 真实模型实验发现，旧 Advisor 能提高市场份额，却在五个共同 Seed 上同时降低利润和企业价值。根因是旧 Advisor 使用价格反应代理分数，不使用真实 `MarketEnv` 长期结果。

Stage 6 不继续增加市场动作，也不立即调用真实模型。第一版先建立三个零 Token 基础：

1. 使用真实市场引擎的长期反事实裁判；
2. 使用 Persona Utility 和风险厌恶的数学行动排序；
3. Known/Mixed/Unknown Holdout 对手预测基准。

## 2. 权威裁判与在线建议必须分开

`AuthoritativeMarketRolloutEvaluator` 从完整冻结 `MarketState` 出发，因此能精确比较候选的真实市场结果。但完整状态包含不完全信息 Agent 不应看到的字段，所以产物固定声明：

- `uses_authoritative_hidden_market_state=true`；
- `allowed_in_public_agent_context=false`；
- `recommendation_is_non_binding=true`；
- `claims_nash_equilibrium=false`。

这使它成为离线 Oracle、Regret 标签和 Advisor 校准器，而不是直接进入公开信息 Agent Prompt 的新建议。后续在线 Advisor 必须从合法 Observation、Belief 和 Opponent Model 构建公开信息预测状态，再用本 Oracle 评估预测误差；不能把隐藏状态通过“建议”间接泄露给 Agent。

## 3. 有限候选集

第一版不搜索连续动作空间，而是从当前约束生成 7–10 个合法候选：保持、小幅降价、大幅降价、提价、增加广告、增加服务、增加产能、增加私人抗冲击投入、开启合作时增加共享抗冲击贡献，以及存在事故时立即维修。

候选先通过原 `ActionValidator`，现金不足、末轮无效投资、关闭合作时的共享贡献和重复动作会被删除。市场公式、安全护栏和真实联合结算均未修改。

## 4. 多轮反事实与 Persona Planning

每个候选默认执行 `5 轮视野 × 10 个确定性情景 × 真实 MarketEnv`。所有候选共享同一组情景 Seed。第一轮对手响应根据公开 Opponent Model 的策略分布和价格反应倾向确定性抽样，后续轮使用现有 Rule Policy 根据分叉后的市场状态继续响应。

每条路径记录最终企业价值、企业价值增量、累计利润增量、事故与缺货机会损失、风险调整价值、最终份额、折扣 Persona Utility 和最坏情况。

Persona Planning Layer 将 Persona Utility 转换为同量纲价值奖励，并使用 `risk_aversion` 在期望人格对齐价值和最坏情况之间计算确定性等价值。由此 Persona 不再只通过 Prompt 改变说法，而会在相同状态、相同候选和相同情景下改变数学排名。

## 5. 对手保留集

新增零模型调用的对手基准：Known 包含增长、利润、防御、合作和平衡；Mixed 包含增长/利润和防御/合作混合；Unknown 包含 100 个由确定性 Dirichlet(1) 生成的效用假设；最后 30 个 Unknown 固定为 Holdout，与开发集完全不重叠。

预测对象是“哪种效用函数更能解释公开行为”，不是对手隐藏 Persona 身份。当前在线 v1 与候选 v2 在保留集上的结果为：

| 指标 | v1 | v2 |
|---|---:|---:|
| Top Strategy Accuracy | 70.00% | 66.67% |
| Distribution Brier | 0.072753 | 0.071301 |
| Cross Entropy | 1.245335 | 1.238481 |

v2 的概率校准和交叉熵较好，但最高类型准确率下降。因此 v2 只保留为实验候选，没有替换现有在线 v1。这是 Stage 6 的首个晋升门禁：不能因为部分指标改善就改变正式 treatment。

## 6. 五 Seed 确定性验收

配置：Seeds 1–5、1 家焦点公司加 3 个 Rule Opponents、balanced market、三种 Persona、5 轮 Rollout、10 情景、8 个基础候选。总计 15 个计划，不调用真实模型。

| Persona | 五 Seed 推荐序列 | 相对 Maintain 平均确定性等价值增益（分） | 推荐动作平均企业价值增量（分） |
|---|---|---:|---:|
| 极端激进 | 广告、服务、抗冲击、服务、服务 | 429,920 | 8,130,465 |
| 短期逐利 | 广告、服务、提价、提价、服务 | 404,368 | 8,186,832 |
| 风险防御 | 抗冲击、抗冲击、抗冲击、服务、抗冲击 | 752,465 | 7,859,216 |

三种 Persona 的 Maintain 平均企业价值增量均为 7,793,755 分；大幅降价仅为 6,284,141 分。该结果说明真实市场 Rollout 能识别旧价格代理容易忽略的价格战损失，并且数学人格层在共同状态下产生稳定但不同的行动排序。

这些是确定性工程与合成市场证据，不是“大模型已经改善”的行为结论。

## 7. 真实模型成本门禁

`RealModelCostGuard` 默认拒绝所有真实 Provider 调用。未来实验每次调用前必须显式授权，并根据固定价格快照预留调用数、输入 Token、输出 Token 和预计费用；调用后写入实际 usage，任一累计指标超过预算立即停止，未预留调用一律拒绝。

本阶段真实模型使用量为：Calls=0、Input Tokens=0、Output Tokens=0、Estimated Cost=0 CNY。

## 8. 最终验收状态

- Stage 6 新增测试：8 项通过；
- 当前全量测试：212 项通过；
- 标准 P0 实验：全部门禁通过；
- 产物：`runs/stage6-strategic-reliability-p0/summary.json`；
- 标准 8 候选 × 10 情景 × 5 轮单计划本机耗时约 0.7 秒。

公开信息在线规划已在 Stage 6.1 完成，Persona/Belief/Opponent Model 下游消融在 Stage 6.2 完成，目标冲突和固定权重失败在 Stage 6.3 完成。Stage 6.4 最终实现约束式 Pareto Planner，修复不公平信息权限与 Regret 候选口径，使用 10 个共同 Seed、720 局完成开发集/保留集验证；三个目标 Persona 均通过企业价值、最坏配对、竞争差距、第一名率、两轮 Regret 和非平凡动作变化门禁，并新增可回放的公开信息 `pareto_rollout_v4`。因此合成市场范围内的 Strategic Decision Reliability v1 已完成，真实 LLM 与现实数据校准仍保持未完成和成本受控状态。详细结果见 `docs/stage6-public-rollout-v1.md`、`docs/stage6.2-strategic-ablation.md`、`docs/stage6.3-objective-calibration.md` 和 `docs/stage6.4-pareto-reliability.md`。

## 9. Stage 6.5 真实 LLM 验证状态

Stage 6.5 已把 Pareto Advice 接入真实 LLM 决策并新增可回放的采纳 Trace。低 Token 渐进实验只在每个 10 轮 Episode 的第 7 轮调用一次 Provider；65 次豆包真实调用、13 个共同配对显示 Pareto 使动作 13/13 改变，推荐采纳率为 84.62%。但短期逐利和风险防御人格仍出现企业价值负尾部，最坏配对分别为 -1,919,523 分和 -638,557 分。

因此 Stage 6.4 的“合成市场 Pareto Planner”结论保持成立，但“Pareto + Belief + Persona 能稳定改善真实 LLM 长期决策”的 Stage 6.5 研究门禁未通过。实验按预注册成本规则停止，没有继续运行 Seeds 6–10 或 Known/Mixed/Holdout 泛化。详细实验、Token、费用与下一修复方向见 `docs/stage6.5-real-advisor-adoption.md`。

## 10. Stage 6.6 失败归因

Stage 6.6 没有修改算法或再次调用真实模型，而是重放 65 个冻结 Episode，并对五个负配对执行“只替换付费轮动作、后续全部动作固定”的真实 MarketEnv 反事实。五个失败决策全部精确执行 Pareto 推荐，Adoption Regret 和 Execution Regret 均为 0；四例主要由 Public Forecast 高估或未覆盖内生对手反应造成，一例由有限候选集合遗漏无建议 Agent 的更优自定义运营组合造成。

这推翻了“当前五个失败首先应通过 Safe-menu Prompt 修复”的假设。Safe-menu 仍可作为未来语义漂移防护，但 P0 应先校准 Forecast、增加低置信度 abstain，并修复候选运营基线覆盖；在这些零 Token 门禁通过前，不运行附件建议的 30 次真实模型接口复测。详细证据见 `docs/stage6.6-failure-forensics.md`。

## 11. Stage 6.6 可靠性修复

法医归因后新增独立 `pareto_reliable_v5`，没有改变已冻结的 v3/v4 行为。v5 为候选集合补入零可选投入 `status_quo`、向上取整到百元的盈亏平衡 `profit_recovery` 和小额韧性投入 `risk_buffer`；再用对手模型置信度、Planner 与退路的价值/尾部/人格比较、候选优势和 Rollout 不确定度决定是否 abstain。Planner 原推荐和门控后的有效推荐分别保存，安全集合与排除集合互斥，并由 Advice Hash 与 Gate Hash 防篡改。

对 Stage 6.5 五个失败冻结状态执行零 Token Auto-execute：修复动作相对同动作带无建议基线为 `+122,032、0、+273,319、0、+54,104` 分，5/5 非负、3/5 严格为正，平均 `+89,891` 分，最差为 0；确定性重建、公开信息边界、动作合法性全部通过。该结果只证明已知失败状态的工程修复，不等于真实 LLM 行为已经改善；下一步仅需对这五个状态做小规模 Safe-menu 付费复测，无需直接运行 30 次。

## 12. Stage 6.7 候选动作与边际投入修复

后续 10 次真实 DeepSeek 配对证明 v5 被 5/5 精确采纳，平均企业价值提高 423,503 分，但风险防御 Seed 4 因 `status_quo` 清零有效的 500,000 分服务投入而出现 −85,845 分负尾部。项目没有修改历史 v5，而是新增版本化 `pareto_reliable_v6`：从零可选投入基线分别测量广告、服务、产能、韧性、共享贡献和维修，再做组合交互门禁；v6 的 `status_quo` 是“边际筛选后的经营基线”。

对同五个真实 LLM Control、同冻结状态和同动作带零 Token Auto-execute，v6 的企业价值差为 `+106,305、+148,600、+625,475、+1,079,787、+306,173` 分，5/5 严格为正，平均 +453,268 分。20 个单项预测与真实冻结轨迹的方向一致率为 70%，说明候选缺口已经修复，但公开 Forecast 仍需校准，尤其是事件下韧性和服务。市场重复结算完全一致，没有发现内部转移错误；现实参数仍因缺少外部数据而未校准。详见 `docs/stage6.7-marginal-candidate-repair.md`。
