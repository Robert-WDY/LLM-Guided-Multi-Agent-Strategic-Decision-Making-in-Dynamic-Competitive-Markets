# Stage 6.6：五个负配对离线法医归因

日期：2026-08-25。

## 1. 本轮边界

本轮严格遵守“先不要修改算法”的要求：没有修改 Pareto Planner、Public Forecast、Prompt、Controller 或 MarketEnv，也没有重新调用真实模型。分析只使用 Stage 6.5 已冻结的 65 条豆包输出和原始 RoundEvent。

目标不是先证明 Advice Contract v2 有效，而是回答五个负配对到底来自 Planner、LLM 采纳、Controller 调整、Forecast，还是候选集合覆盖不足。

## 2. 归因口径

每个共同状态固定第 7 轮前的真实 `MarketState`。只替换焦点公司第 7 轮动作，第 8–10 轮焦点公司与三个对手的已结算动作全部冻结为原 Pareto Episode 的动作带，MarketEnv 使用原 Episode Seed 和组件随机源重新结算。

这会得到三个可加总的 Regret：

```text
Planner Regret   = V(候选集合内真实最优) - V(Advisor 推荐)
Adoption Regret  = V(Advisor 推荐) - V(LLM 请求)
Execution Regret = V(LLM 请求) - V(Controller 最终动作)
```

三项之和严格等于候选 Oracle 到最终动作的差。Forecast Error 是重叠诊断项，不能再次加入总和：它比较公开预测的企业价值和相同规划视野下的真实企业价值。另记录“内生反应路径效应”，即原始条件差减去固定动作带条件差，用来识别后续对手反应是否改变结论。

若无建议 Agent 的自定义动作优于候选集合中的真实最优动作，则额外记录 Candidate Set Coverage Regret。这个诊断是必要的，因为有限候选集合本身可能遗漏更好的经营组合。

## 3. 五个失败配对

金额单位均为分。Forecast Error 为正表示 Planner 高估，内生路径效应为负表示 Pareto 路径触发的后续反应进一步降低相对价值。

| Seed | Persona | 推荐 | 实际候选 Oracle | 原始价值差 | 固定动作带价值差 | Planner Regret | Adoption | Execution | Forecast Error | 内生路径效应 | 最大来源 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 风险防御 | maintain | price_increase | -80,452 | +352,833 | 224,694 | 0 | 0 | +163,294 | -433,285 | Forecast/后续反应 |
| 2 | 短期逐利 | price_cut_small | increase_resilience | -1,919,523 | -217,581 | 193,167 | 0 | 0 | +5,304,504 | -1,701,942 | Forecast 高估 |
| 3 | 风险防御 | price_increase | increase_resilience | -638,557 | -697,787 | 900,747 | 0 | 0 | +4,555,677 | +59,230 | Forecast 高估 |
| 4 | 短期逐利 | maintain | maintain | -825,357 | -800,765 | 0 | 0 | 0 | -1,160,924 | -24,592 | 候选集合覆盖不足 |
| 4 | 风险防御 | maintain | price_increase | -57,225 | +122,601 | 55,194 | 0 | 0 | -4,792,691 | -179,826 | Forecast/后续反应 |

五个状态中：

- 5/5 都是 `exact_action`，LLM 请求动作与推荐候选完全一致；
- 5/5 的请求动作和 Controller 最终经济动作一致；
- Adoption Regret 5/5 为 0；
- Execution Regret 5/5 为 0；
- 四例的主要问题属于 Forecast 误差或没有覆盖内生对手反应；
- 一例属于候选集合覆盖不足：Seed 4 短期逐利中，Advisor 已选中候选集合内真实最优 `maintain`，但该候选带有公开运营基线的广告、服务和产能投入，无建议 Agent 的自定义动作仍高 800,765 分。

因此，这五个失败不能归结为“LLM 不会使用建议”。模型恰恰非常忠实地执行了建议，而建议所依据的预测或候选集合不够可靠。

## 4. 逐轮证据

完整的五个冻结状态、候选集合、安全集合、Planner 推荐、LLM 原始 JSON、Requested Action、Final Action、每轮三个对手动作、每轮随机数摘要及第 7–10 轮 MarketEnv 结果保存在：

- `runs/stage6.6-failure-forensics/failure-cases.json`；
- `runs/stage6.6-failure-forensics/historical-mapping.json`；
- `runs/stage6.6-failure-forensics/replay-checks.json`；
- `runs/stage6.6-failure-forensics/summary.json`。

反事实使用修改后的 Action ID，因此原始 State Hash 会改变；排除 Action ID、Event ID 等非经济字段后，5/5 原 Pareto 经济路径完全复现，最终企业价值也与 Stage 6.5 行记录一致。

## 5. 65 条冻结输出映射

65 条历史输出全部被映射为“精确候选”或明确的 `custom_action`，没有用最近候选伪装自由数值动作：

- 精确 Requested Candidate：12/65；
- 精确 Final Candidate：12/65；
- Final 位于 Pareto Safe Set：11/65；
- 自定义 Final Action：53/65；
- Pareto v4 条件自身为 12/13 精确候选、11/13 位于 Safe Set；
- 五个负配对全部属于精确且安全的推荐候选；
- 65 条中只有一条非 Pareto 基线存在 Controller 经济调整，五个负配对没有调整。

这说明 Safe-menu 对约束一般自由输出仍可能有价值，但它不能修复本次五个失败，因为这些失败已经处于 Safe Set 内并被精确执行。

## 6. Replay 与泄漏验收

65 个 Episode 全部通过：

- Economic Replay 100%；
- Interaction Replay 100%；
- Information Replay 100%；
- Belief Replay 100%；
- Advisor/GameTheory Replay 100%；
- Adoption Trace 重建与 Hash 匹配 100%；
- 五个 Pareto 经济路径复现 100%；
- Final Illegal Action = 0；
- Hidden State Leakage = 0；
- 真实模型调用 = 0。

本轮没有伪造“新 Advice Contract Hash 100%”。Advice Contract v2 尚未实现，状态明确记录为 `deferred_until_forensic_decision`。

## 7. 对下一步的修正

附件原建议把 Safe-menu Advice v2 作为下一实现，但法医证据表明它不是当前五个失败的首要修复。直接做 30 次 Free-form vs Safe-menu 真实调用，只会重复验证已经为 0 的 Adoption/Execution Regret，无法解决 Forecast 和候选覆盖问题。

下一步 P0 应先保持零 Token：

1. 为 Public Forecast 建立同视野校准表，分别记录预测高估、低估和内生对手反应误差；
2. 让 Planner 在 Forecast 历史误差超过 Persona 容忍值时 `abstain → maintain/Rule baseline`；
3. 区分“候选 maintain”和“无建议运营基线”，补足 Seed 4 短期逐利暴露的候选覆盖缺口；
4. 在五个冻结状态执行 Auto-execute Control，要求推荐或 abstain 相对无建议基线不产生负尾部；
5. 上述门禁通过后，再实现 Safe-menu Contract v2，并把它作为语义漂移防护，而不是把它当作本轮失败的根因修复。

正式汇总文件 SHA-256 为 `1930BD7663A5E43A1F78005EC5D38D53CEB8C9E6322BD220DD28B0C234DAD196`，内部 `report_hash` 为 `sha256:23ba73aba695ce386996e4da36e5cc23feb09250de99558fe3bed99a89fcb68f`。

## 8. 根因修复与零 Token 验收

法医门禁通过后，新增独立 `pareto_reliable_v5`，不覆盖旧 v3/v4 合同和哈希。修复包括：

1. 候选集合增加零可选投入的 `status_quo`、向上取整到百元的盈亏平衡 `profit_recovery`、以及小额韧性投入 `risk_buffer`；等价动作会确定性去重；
2. 可靠性门控综合对手模型置信度、Planner 相对情境退路的预期价值/最坏价值/人格价值、候选优势和 Rollout 情景范围；证据不足时明确 abstain；
3. abstain 后按公开决策支持和事件选择情境退路，不再用已经被判定不可靠的同一 Forecast 把退路循环改回旧 Planner 动作；
4. Advice 同时保留 Planner 原推荐和有效推荐，安全候选与排除候选互斥，Gate Hash 和 Advice Hash 均可检测篡改。

五个失败状态使用原豆包输出、相同 MarketState、相同随机源和相同后续动作带 Auto-execute，结果如下：

| Seed | Persona | v5 有效动作 | 相对无建议基线企业价值差（分） |
|---:|---|---|---:|
| 1 | 风险防御 | status_quo | +122,032 |
| 2 | 短期逐利 | status_quo | 0 |
| 3 | 风险防御 | risk_buffer | +273,319 |
| 4 | 短期逐利 | profit_recovery | 0 |
| 4 | 风险防御 | status_quo | +54,104 |

门禁结果为 5/5 非负、3/5 严格正、平均 `+89,891` 分、最差 0；5/5 都触发 abstain，5/5 确定性重建，5/5 只使用公开与本公司私有输入，Final Illegal Action 为 0。新增真实模型调用、输入 Token、输出 Token 和费用全部为 0。新增范围 Ruff、`git diff --check` 和全量 220 项测试通过；全项目 Ruff 仍报告若干旧实验脚本的顶层导入顺序遗留问题，本阶段没有扩大范围修改。证据保存在 `runs/stage6.6-reliable-repair/summary.json`，文件 SHA-256 为 `00AD2B94075229981AEC49F788950869E8939F20616C7D4FABDEA17F44F9E30C`，内部 Report Hash 为 `sha256:3413f00535b6b7d6d74b9e3c24c0daf346a4a0955530102a23b250e832bb81c2`。

结论边界：这是对五个已知失败状态的冻结动作带修复验收，尚未证明真实 LLM 会正确理解并使用 v5。下一次付费实验应只复测这五个状态，并设置严格调用/Token 上限；在此之前不扩大到 30 次。
