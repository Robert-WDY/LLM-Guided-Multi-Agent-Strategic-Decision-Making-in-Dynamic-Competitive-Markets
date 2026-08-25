# Stage 6.1：公开信息在线战略规划 v1

日期：2026-08-24。

## 1. 目标与安全边界

Stage 6 P0 的长期 `AuthoritativeMarketRolloutEvaluator` 使用完整 `MarketState`，只能作为离线 Oracle。Stage 6.1 新增 `public_rollout_v3`，目标是在不完全信息市场中把长期规划安全地送入 Agent 决策上下文。

在线规划器只读取：公共市场状态、本公司完整私有状态、公开价格信念、公开对手策略模型、当前 Persona Profile 和版本化市场配置。它不读取对手现金、成本、产能、品牌内部资产、私人抗冲击能力、事故详情、Persona、真实效用或未来随机数。

真实 `state_hash` 和外层 `observation_hash` 只负责观察绑定，不能选择预测情景。预测随机 Seed 由净化后的合法决策输入计算，避免隐藏状态通过 TrueState Hash 间接改变建议。

## 2. 公开预测市场

`build_public_forecast_state()` 把合法 Observation 重建为可运行的预测市场：

- 本公司使用自己的完整 `PrivateState`；
- 对手只覆盖公开价格、份额、销量和声誉；
- 对手不可见字段使用配置中性值，Persona 固定为 `none`，私人事故固定未知；
- 已公开事件类型和严重度通过版本化配置还原经济影响；
- 已公开销量是对手估计有效产能的下限，避免产生“销量超过预测产能”的不可能状态；
- Belief 与 Opponent Model 必须与 Episode、观察公司、轮次和 State Version 完全绑定。

契约固定声明 `uses_authoritative_hidden_market_state=false`、`uses_hidden_opponent_state=false`、`hidden_opponent_fields_are_estimated=true` 和 `allowed_in_public_agent_context=true`。

## 3. 在线候选必须保留正常运营

第一次 5 Seed 试验暴露出一个重要设计错误：P0 Oracle 的候选为隔离因果维度，会把未研究的可选投入设为 0；如果直接执行“增加服务”，正常的广告或其他运营预算会被清空。第一轮结果虽然平均企业价值增加 961,406 分、平均权威遗憾下降 381,330 分，但 Seed 1、2 的最终企业价值分别低于 Rule 基线 2,809,037 和 3,905,545 分，因此没有晋升。

修复后新增 `generate_public_overlay_candidates()`：先用确定性 Rule Policy 形成合法运营基线，再对价格、广告、服务、产能或抗冲击能力只施加一个战略增量。`maintain` 表示保留整套运营基线，不再表示“除价格外所有预算归零”；价格候选也保留基线广告和服务。这一修复没有改变市场公式，只修正了在线候选的动作语义。

## 4. API、上下文与回放

Episode 可配置 `advisor_mode=public_rollout_v3`。该模式强制要求：

- `information_mode=public`；
- `belief_mode` 已开启；
- `opponent_model_mode=public_strategy_v1`；
- 不允许任一公司覆盖为 `perfect`。

建议写入原 `game_theory_advice` 字段并由 `Observation Hash` 覆盖，进入 `DecisionContext`。Prompt 明确说明预测市场的估计边界、非绑定性质和非 Nash 性质。Advisor Replay 会从原 Observation、Persona、Belief 和 Opponent Model 重新生成 Advice，并验证 Advice Hash 与完整内容。

`MockModelClient(honor_game_theory_advice=true)` 已支持采纳 v3 的完整结构化动作，而不再只读取旧 Advisor 的推荐价格；统一 Planner 和 Controller 安全约束仍可缩减非法或超预算请求。`four_agent_acceptance` 同步开放 v3 作为零模型消融选项。

默认在线规格为 `3 轮视野 × 5 情景`；确定性验收为了同时计算在线建议与离线 Oracle，使用 `3 轮 × 3 情景`。

## 5. 五共同 Seed × 10 轮 Rule 对照

配置：balanced market、balanced_v1 Persona、1 个公开信息规划器、3 个 Rule 对手、Seeds 1–5、每个 Episode 10 轮。Baseline 为 4 个 Rule；Treatment 只将 company_A 替换为每轮重新规划的 Public Rollout Policy。两个条件共享市场 Seed。真实模型使用量为 0。

| Seed | 企业价值配对差（分） | 累计利润配对差（分） | 在线建议平均 Oracle Regret（分） | Maintain 平均 Oracle Regret（分） |
|---:|---:|---:|---:|---:|
| 1 | +10,028,891 | +11,988,960 | 62,632 | 1,125,162 |
| 2 | +2,405,123 | +2,439,454 | 0 | 188,200 |
| 3 | +9,778,693 | +9,941,298 | 345,943 | 1,373,890 |
| 4 | +11,282,858 | +12,578,310 | 51,301 | 714,883 |
| 5 | +5,794,188 | +5,371,890 | 499,461 | 899,307 |

五 Seed 平均企业价值提高 7,857,951 分，平均累计利润提高 8,463,982 分；平均 Oracle Regret 从 860,288 分降至 191,867 分，下降 668,421 分。五个 Seed 的企业价值配对差全部非负，因此三项策略晋升门禁全部通过。

该结果允许 `public_rollout_v3` 进入 Mock/Rule 研究 treatment，不证明真实 LLM 会采纳建议，也不是统计显著性结论。

## 6. 验收与成本

新增 8 项测试覆盖公开预测状态、隐藏对手字段非干扰、运营基线保留、确定性、Advice Hash 篡改、Perfect 模式拒绝、API Observation、Advisor Replay 和小型 Rule 验收。全量 201 项测试通过，`git diff --check` 通过。

标准产物：`runs/stage6-public-rollout-v1/summary.json`。

真实模型使用：Calls=0、Prompt Tokens=0、Completion Tokens=0、Estimated Cost=0 CNY。

## 7. 下一步

上述零模型消融已在 Stage 6.2 完成。Persona Planner 在 27/27 配对中提高企业价值，但极端激进人格使一步 Regret 上升；Belief 降低一步 Regret 却降低五轮价值；Opponent v1 有局部收益但存在尾部退化；v2 为 0 组改善、3 组恶化，明确不晋升。实验同时修复了不同 treatment 错用不同 Advisor 随机场景的混杂问题。完整结果见 `docs/stage6.2-strategic-ablation.md`。下一步先做目标与标签校准，不启动真实模型。
