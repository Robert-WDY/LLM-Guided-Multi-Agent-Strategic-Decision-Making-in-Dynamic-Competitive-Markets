# Game Theory Enhancement：实现与验收

## 1. 阶段边界

本阶段把原来的“预测对手下一轮价格方向”扩展成下面的可审计链路：

```text
Public Observation / Settled Public History
  -> Opponent Strategy Model
  -> Opponent Utility Inference
  -> Expected Opponent Response
  -> Approximate Bayesian Best Response Advisor
  -> Agent Private Decision
  -> Final Joint Action
  -> Market Settlement
```

合作历史还会进入独立的重复博弈建议：Tit-for-Tat、Grim Trigger 和 Generous Tit-for-Tat。

所有新增模块只提供信息、概率或非绑定建议，不直接修改 `MarketState`，不执行动作，也不读取对手现金、真实成本、Persona 或 Prompt。本阶段没有求 Bayesian Nash Equilibrium。

## 2. Phase 1：Opponent Modeling

新增 `opponent/` 包：

- `PublicStrategyEvidence` 只记录公开价格、份额变化、销量、声誉和公开共享韧性贡献；
- `OpponentBehaviorProfile` 输出价格进攻性、公开扩张性、风险容忍和合作倾向；
- `StrategyDistribution` 输出 `growth / profit / defensive / cooperative` 的整数概率；
- `OpponentModelLedger` 在每次结算后增加公开证据，同轮重复写入必须完全相同；
- `verify_opponent_model_replay` 从初始状态、公开联合动作和结算状态重建每个公司实际收到的模型与 Hash。

v1 使用确定性 Rule-Bayesian 计分和带平滑先验的归一化，不调用 LLM。它输出的是可检验的类型假设，而不是真实 Persona 标签。

## 3. Phase 2：Utility Inference

新增 `utility_inference/` 包。它把对手策略概率映射成六项效用权重分布：

- Profit；
- Market Share；
- Risk Avoidance；
- Cash Preservation；
- Growth；
- Social Welfare。

所有均值严格合计 `1,000,000 ppm`，置信度来自公开证据数量。每个推断记录绑定源 `OpponentModel Hash`，`verify_utility_inference_replay` 会重新计算并拒绝不一致数据。`uses_hidden_persona/profit/cash` 均由契约固定为 `false`。

这是一种“哪些效用更能解释公开行为”的代理模型，不是对真实内在动机的直接观测。

## 4. Phase 3：Game Theory Advisor v2

新增 `advisor_mode=bayesian_strategy_v2`。它需要：

- `belief_mode != off`；
- `opponent_model_mode=public_strategy_v1`；
- `utility_inference_mode=strategy_utility_v1`。

Advisor 在有限价格动作上比较：激进降价、普通降价、维持、涨价。对每个候选动作，它计算：

- 对手跟随降价/维持/涨价的预测概率；
- 预期利润代理；
- 预期市场份额代理；
- 战略风险；
- Expected Utility Proxy；
- Worst-case Utility Proxy。

最终输出 `recommended_action` 和 `recommended_price_cents`，但固定：

- `recommendation_is_non_binding=true`；
- `approximate_best_response=true`；
- `claims_nash_equilibrium=false`；
- `uses_hidden_opponent_state=false`。

`MockModelClient(honor_game_theory_advice=True)` 仅用于证明“Advice 能进入最终决策”的确定性管线。真实模型仍可以拒绝建议。

## 5. Phase 4：Repeated Game Strategy

新增 `repeated_game/` 包和 `repeated_game_mode=reciprocity_v1`。它只从权威 `cooperation_memory` 派生：

- Tit-for-Tat stance；
- Grim Trigger stance；
- Generous Tit-for-Tat stance；
- 当前推荐 stance；
- 建议贡献倍率。

一次偶发违约可进入谨慎或反制；两次明确背叛会触发永久拒绝。建议本身不改变市场，实际共享韧性贡献仍必须由 Agent Action 提交。

## 6. Agent、API、日志与 Replay 集成

Episode 新增四个显式 treatment：

```json
{
  "opponent_model_mode": "public_strategy_v1",
  "utility_inference_mode": "strategy_utility_v1",
  "advisor_mode": "bayesian_strategy_v2",
  "repeated_game_mode": "reciprocity_v1"
}
```

Observation/DecisionContext 新增对应 State 与 Hash。Planner Prompt 明确区分公开事实、策略概率、推断效用和非绑定建议。RoundEvent v1.9 记录：

- `belief_before`；
- `opponent_model`；
- `utility_inference`；
- `advisor_output`；
- `repeated_game_strategy`；
- `chosen_action`；
- `counterfactual_results`；
- `belief_after`。

新增 `verify_game_theory_replay`，组合验证 Opponent、Utility、Advisor、Repeated Game 重建，并核对 Trace 实际输入、候选反事实和 Final Action 绑定。Economic、Interaction、Information 和 Belief Replay 的原有职责保持不变。

## 7. 实现中发现并修复的问题

### 7.1 Communication Context 与 Decision Context 不是同一输入集合

首轮 4-Agent 验收中，Economic、Interaction 和 GameTheory Replay 通过，但 Information Replay 报告：Communication Context 缢少 `game_theory_advice`。原因是 Information Replay 误把用于生成 cheap-talk 消息的较小 Communication Context，当成完整私人经济 Decision Context 验收。

修复后：Decision Context 仍必须完整绑定四个战略字段；Communication Context 只校验其契约实际包含的字段。没有把 Advisor 扩散到不需要它的通信 Prompt，也没有放松经济决策校验。

### 7.2 Belief Replay 的计数口径重复

验收脚本最初按“通信生成一次 + 私人决策一次”预期每公司每轮两条 Belief Replay 结果，但 Belief Replay 会对同公司同轮去重，因此实际是每公司每轮一条。修复的是验收计数，不是 Replay 数据。

### 7.3 本机 `.env` 污染负向 API 测试

旧测试期望未配置 Controller Token 时返回 503，但工作区 `.env` 已配置开发 Token，导致返回 401。测试现在先显式删除该环境变量，再验证 Disabled 分支，然后设置测试 Token 验证授权分支。生产逻辑没有改变。

### 7.4 版本语义同步

新增战略字段后，Decision Context/Planner Prompt 默认版本升至 v1.14，RoundEvent 升至 v1.9，Episode Manifest 升至 v1.6。旧版本仍可解析；测试断言同步到新默认，避免同一版本代表不同输入语义。

### 7.5 只开启 Opponent Model 时也必须保护 PrivateState

集成复核发现：Episode Creation 已把 Opponent/Utility/Advisor/Repeated treatment 视为受保护创建并返回公司令牌，但旧 Gateway 鉴权的“兼容免令牌”判断只检查 Communication、Cooperation 和 Belief。单独开启 `public_strategy_v1` 时，调用者可能无需令牌读取某公司的 Observation，其中包含该公司的 PrivateState。

修复为：任何战略 treatment 开启时都必须校验 company-scoped Agent Token；无令牌、使用其他公司令牌均返回 401。新增专门回归证明只有 company_A 的令牌可以读取 company_A 的私有视图。

## 8. 确定性 4-Agent 闭环验收

命令：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.experiments.four_agent_acceptance `
  --episode-id game-theory-enhancement-acceptance-v2 `
  --seed 20260821 --rounds 5 --market-model balanced `
  --information-mode public `
  --communication-mode public_private `
  --cooperation-mode shared_resilience_v1 `
  --belief-mode public_action_signal_v2 `
  --opponent-model-mode public_strategy_v1 `
  --utility-inference-mode strategy_utility_v1 `
  --advisor-mode bayesian_strategy_v2 `
  --repeated-game-mode reciprocity_v1 `
  --honor-game-theory-advice `
  --provider mock --llm-count 4 `
  --output runs/game-theory-enhancement-acceptance-v2-20260821
```

结果：

- 5 轮、4 Agent、20 次私人决策全部完成；
- LLM/Mock fallback 0；
- 每轮一次结算、重复 Action 0、部分状态更新 0；
- Economic Replay 100%；
- Interaction Replay 100%；
- Information Replay 100%；
- Belief Replay 100%；
- GameTheory Replay 100%；
- Opponent Model View 20、Utility View 20、Advisor View 40、Repeated Game View 20；
- Strategic Trace Binding 80；
- Hidden State Leakage 0。

正式产物：`runs/game-theory-enhancement-acceptance-v2-20260821/summary.json`。

## 9. 合成行为基准

命令：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.experiments.game_theory_enhancement_acceptance `
  --closed-loop-summary runs/game-theory-enhancement-acceptance-v2-20260821/summary.json `
  --output runs/game-theory-enhancement-benchmark-20260821
```

四种隐藏生成策略各生成 10 轮公开证据。结果：

| 指标 | Opponent Model | Price-frequency baseline |
|---|---:|---:|
| Type Accuracy | 100% | 75% |
| Multiclass Brier（越低越好） | 0.1502 | 0.4200 |
| Log Loss（越低越好） | 0.3339 | 0.8432 |

增长型场景中，No-Advisor 的 `maintain` 相对 Advisor 推荐的内部 Expected Utility Proxy Regret 为 `15,808,300`；Advisor 自己选择候选集最大值，因此同口径 Regret 为 0。面对增长型对手推荐 `aggressive_price_cut`，面对防御型对手推荐 `maintain`，满足战略适应门禁。

重复博弈反事实中，高信誉且三次履约的对手得到 `cooperate / 1,000,000 ppm`；低信誉且两次背叛的对手得到 `permanent_refusal / 0 ppm`。

正式产物：`runs/game-theory-enhancement-benchmark-20260821/summary.json`。

## 10. 证据边界与下一步

本阶段可以说：

- Agent 输入中存在显式、可回放的对手策略和效用假设；
- Opponent Modeling 在四种合成生成过程上优于只看价格方向的基线；
- Advisor 扩展了有限动作搜索，并在至少一个合成场景降低内部 Proxy Regret；
- 相同 Agent 面对增长型和防御型对手得到不同建议；
- 履约/背叛历史能确定性改变重复博弈建议；
- 完整链路没有读取对手隐藏现金、成本、Persona 或 Prompt。

不能说：

- 找到了 Nash 或 Bayesian Nash Equilibrium；
- 真实 LLM 已理解对手动机；
- Advisor 总能提高实际利润；
- 四个手工构造类型的 100% Accuracy 能外推到自然市场；
- Utility Weight 已恢复真实心理偏好。

下一步应冻结 v1 工程接口，用共同 Seed 做 `Persona only / +Belief / +Opponent Model / +Utility / +Advisor` 配对实验；增加真实模型固定状态重复、same-treatment placebo、未知/混合类型和 Holdout 对手，并把实际 Market Profit、Risk Loss、Share 与内部 Proxy Regret 分开报告。
