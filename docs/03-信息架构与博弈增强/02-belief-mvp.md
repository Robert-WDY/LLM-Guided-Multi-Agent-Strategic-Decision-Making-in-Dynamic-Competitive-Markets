# Belief MVP（Phase B）

## 目标与边界

本阶段把 Phase A 的“可见/不可见信息”升级为可回放的简单对手信念：

```text
已结算公开价格历史
  -> PublicPriceEvidence
  -> company-scoped BeliefLedger
  -> 对手下一轮 price_cut / maintain / price_raise 概率
  -> Belief Hash
  -> Agent Communication/Decision Context
  -> 结算后评分与 Belief Replay
```

这不是完整 Bayesian Best Response。系统不估计对手现金、Persona、计划、成本或效用函数，也不让 LLM 自行维护不可回放的概率。`belief_mode=off` 仍是默认基线；`public_action_v1` 是独立实验处理，与 `information_mode=perfect/public` 正交。

## 确定性更新规则

每个对手有三个类别：

- `price_cut`：本轮最终公开价格低于结算前公开价格；
- `maintain`：二者相同；
- `price_raise`：本轮最终公开价格更高。

每类使用 1 个 Dirichlet 伪计数。第一轮没有历史证据，因此概率固定为 `333334 / 333333 / 333333 ppm`。第 (t) 轮 Observation 只使用第 (1..t-1) 轮证据预测本轮，当前轮最终动作在结算后才更新第 (t+1) 轮信念，避免数据穿越。

概率用整数 ppm 表示，并通过固定方向顺序的 largest-remainder 分配保证总和严格为 1,000,000。相同 Episode、公开动作和顺序会生成完全相同的 Belief State 与 Hash。

## 契约

启用后 Observation 包含：

```json
{
  "belief_schema_version": "belief-state-v1.0.0",
  "belief_hash": "sha256:...",
  "belief_state": {
    "belief_mode": "public_action_v1",
    "updater_version": "dirichlet-public-price-v1.0.0",
    "observer_company_id": "company_A",
    "prediction_target_round": 3,
    "public_evidence_through_round": 2,
    "evidence_scope": "settled_public_prices_only",
    "opponent_beliefs": {
      "company_B": {
        "evidence_count": 2,
        "next_price_direction": {
          "price_cut_ppm": 600000,
          "maintain_ppm": 200000,
          "price_raise_ppm": 200000
        }
      }
    }
  }
}
```

Belief 不包含观察者自己，也不包含任何对手财务、运营、事故、Persona 或计划。`observation_hash` 覆盖完整 Belief；`belief_hash` 使用独立 `belief-view-hash-v1.0.0`，便于单独回放和定位错误。

Episode Manifest v1.4 记录：

- `belief_mode`；
- `belief_schema_version`；
- `belief_updater_version`；
- `belief_hash_protocol_version`。

Agent Observation、Communication Context、Decision Context、Planner Prompt 和 RoundEvent 分别升级到 v1.8、v1.6、v1.12、v1.12 和 v1.8。Prompt 明确：概率是 Controller 根据公开历史计算的不确定估计，不是事实、承诺或指令，不得反推出隐藏状态。

## Replay 与指标

`verify_belief_replay(events, manifest)` 从 round 1 开始：

1. 用初始中性先验重建每个公司本轮输入；
2. 比较记录的 Belief State 和 Belief Hash；
3. 验证同一公司通信与决策阶段的信念一致；
4. 使用 RoundEvent 的 `state_before + joint_action` 生成本轮公开价格证据；
5. 更新 Ledger 后重建下一轮；
6. 拒绝缺轮、错误公司、错误概率、错误证据数或篡改历史。

Information Replay 继续负责 True State→可见字段；Belief Replay 单独负责公开证据→概率。两者分层，避免 Information Replay 信任日志里已经写好的概率。

`compute_belief_calibration` 以“决策前概率、结算后实际方向”计算：Top-1 Accuracy、Multiclass Brier Score 和 Log Loss。统计单位可以继续聚合到 Episode/Seed；不能把同一回合中 12 个观察者-对手记录当成 12 个独立 Seed。

## 验收结果

产物：

- `runs/belief-mvp-p0-acceptance-20260821-v2/summary.json`；
- `runs/belief-mvp-p0-acceptance-20260821-v2/round-events.jsonl`。
- `runs/belief-mvp-p0-off-paired-20260821/summary.json`（相同 Episode ID/Seed 的 OFF 对照）。

配置为 4 Mock、5 轮、共同 Seed `20260821`、`information_mode=public`、`belief_mode=public_action_v1`、通信关闭。结果：

- 20/20 决策成功，0 fallback；
- 5/5 轮单次结算，无部分状态更新；
- Economic Replay 100%；
- Interaction Replay 100%；
- Information Replay 100%；
- Belief Replay 100%；
- 60 个观察者→对手预测，Top-1 Accuracy 80%；
- Mean Brier Score `0.31415648027240006`；
- Mean Log Loss `0.6089045075448257`。

80% 主要来自该确定性 Mock 路径中“维持价格”占优，只能说明更新器能从稳定公开行为形成更集中的可用预测，不能说明真实 LLM 已提升利润或具备完整对手建模能力。

另有同一冻结状态反事实：同一个确定性 Mock Policy 在 `belief_state=null` 时维持原价，在看到达到阈值的公开降价信念时降低报价。该测试证明 `Belief -> Context -> Decision` 管线能够产生动作差异；它是管线证据，不是真实模型效应证据。

默认 Mock 不主动使用 Belief 时，有效配对的 ON/OFF 两组最终 State Hash 都为 `sha256:f6259ed86cbd21d0daec632c506accdca7ccb3d88fcbca4464d9c5ca7b558894`，市场利润也完全相同。这证明 Belief 是输入而不是隐藏的市场干预。第一次 OFF 尝试误用了不同 Episode ID，虽然 Seed 相同，但随机组件键不同，因而市场路径不同；该目录已写入 `INVALID_COMPARISON.md`，正式结论只使用同 Episode ID 的配对产物。

初版 ON 产物随后暴露一个审计字段不一致：顶层已是 `belief-state-v1.0.0`，嵌入 Visibility Policy 仍写 `none`。修复后重新生成 v2；旧目录保留 `SUPERSEDED.md` 只作过程记录。

## 后续实现状态

Belief-aware Planner 的首轮真实模型配对 Pilot 已完成，结果见 `docs/03-信息架构与博弈增强/03-belief-real-paired-pilot.md`。之后已实现 `public_action_signal_v2` 与非绑定 `bayesian_price_v1` Advisor，工程设计和验收见 `docs/03-信息架构与博弈增强/04-incomplete-information-p0-p5.md`。Pilot 仍只表明模型能明确引用信念并在固定状态下方向一致地调整价格；当前预测器信息增益弱、利润方向不稳定，新增 P4/P5 工程能力不能替代扩大真实模型实验。

下一阶段仍不增加更多隐藏字段，而是扩大并强化以下实验：

1. 固定状态至少 10 对，并增加 same-treatment placebo；
2. `neutral / high-cut / high-maintain / high-raise` 概率处理；
3. 多轮至少 10 个共同 Seed；
4. 比较累计频率、滑动窗口与遗忘因子；
5. 通过后再加入通信信号可信度、对手类型信念或 Approximate Best Response。
