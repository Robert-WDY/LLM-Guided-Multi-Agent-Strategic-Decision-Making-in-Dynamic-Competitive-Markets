# ARIN7600 人工智能项目完整报告 v3.0
## 面向不完全信息重复博弈的 LLM 多智能体战略决策系统

**Engineering MVP 规范：** `02-LLM多智能体生鲜配送市场博弈系统_Engineering_MVP技术规格_v4.0.md`
**权威市场配置：** `market_v4.yaml`

---

# 0. 版本说明与唯一规范来源

本报告不是对旧完整文档的局部增补，而是一次结构化重写。

它统一了：

- 动态公司状态；
- 连续 Numeric Action；
- 三类消费者和 Outside Option；
- 现金、产能和终局规则；
- 风险信号、重大事件和公司事故；
- Seeded Randomness；
- Action Idempotency；
- EpisodeManifest；
- 新数据集 Schema；
- Engineering MVP 与 Research MVP 的边界。

规范优先级：

```text
market_v4.yaml
→ Engineering MVP v4.0
→ 本完整报告 v3.0
→ 旧文档仅作历史参考
```

旧英文完整版仍对应旧市场定义，不再作为当前系统实现依据。需要英文提交时，应以本 v3.0 为源重新翻译，而不是继续使用旧英文版。

---

# 1. 项目摘要

本项目研究：

> LLM 能否在一个动态、随机、重复、多智能体且逐步扩展为不完全信息的市场环境中，形成稳健、可解释并具有博弈意识的公司战略？

市场场景为：

> 本地线上生鲜配送平台竞争。

每个 LLM Agent 扮演一家公司，需要根据：

- 自己的 Cash、Profit、Capacity、Brand、Service、Reputation、Resilience；
- 竞争者状态；
- 消费者结构；
- 风险信号；
- 市场事件；
- 公司事故；

决定：

- Price；
- Advertising；
- Service；
- Capacity Investment；
- Resilience Investment；
- Incident Repair；
- 后续 Research 阶段的 Communication / Cooperation。

项目不把 LLM 生成的解释当作战略能力证据，而使用：

- Profit / Market Share；
- Opponent Prediction；
- Approximate Regret；
- Cross-play；
- Welfare；
- Event Response；
- Generalization；

验证 Agent 是否真正改善决策。

---

# 2. 项目阶段命名

为避免“MVP”含义冲突，统一使用两层定义。

## 2.1 Engineering MVP

解决：

> 系统能否正确、稳定、可复现地运行？

包含：

- Dynamic Market；
- Numeric Actions；
- Company State；
- Risk / Incident；
- Joint Action；
- Validator；
- Event Log；
- Replay；
- Rule / Mock / Basic LLM。

不包含：

- Belief；
- Communication Effect；
- Game Theory Advisor；
- PSRO。

## 2.2 Research MVP

解决：

> LLM 的 Persona、信息结构、交流和博弈论模块是否产生可测量的战略影响？

包含：

- Persona Ablation；
- Perfect vs Imperfect Information；
- Belief / Opponent Modeling；
- Communication / Cooperation；
- Welfare；
- Game Theory Advisor；
- Cross-play。

## 2.3 Advanced Research

包含：

- Self-play；
- Approximate Best Response；
- PSRO；
- Policy Population；
- Dataset Training。

---

# 3. 核心研究问题

## RQ1：人格是否影响策略？

固定：

- State；
- Seed；
- Model；
- Prompt Template。

只改变 Persona，比较：

- Numeric Actions；
- Profit；
- Share；
- Risk Preparation；
- Repair Decision；
- Long-term Enterprise Value。

## RQ2：完全信息与不完全信息有什么差异？

比较：

```text
Full State
vs
Agent-specific Observation + Hidden State
```

研究：

- 决策质量；
- Opponent Prediction；
- 风险；
- Cooperation；
- Regret。

## RQ3：Communication 是否改变合作与竞争？

比较：

```text
No Communication
vs
Cheap Talk
vs
Structured Proposal / Commitment
```

## RQ4：公司利益与社会福利是否冲突？

研究：

- 企业 Profit；
- Consumer Utility；
- No-purchase；
- Lost Demand；
- Stockout；
- HHI；
- Waste；
- Market Stability。

## RQ5：Game Theory Advisor 是否帮助 Agent？

比较：

```text
Persona-only
vs
Persona + Belief
vs
Persona + Belief + Game Theory Advisor
```

## RQ6：交互数据能否训练更好的 Policy？

使用：

```text
Observation
+ Action Constraints
+ Numeric Action
+ Utility / Reward
+ Next Observation
+ Event Context
```

训练或蒸馏小型 Policy，并做未见对手测试。

---

# 4. 形式化定义

最终环境建模为有限时域部分可观测随机博弈：

\[
G=\langle N,S,\{A_i\},T,\{O_i\},Z,\{U_i\},E,H\rangle
\]

其中：

- \(N\)：公司 Agent；
- \(S\)：真实市场和公司状态；
- \(A_i\)：连续、有约束的公司动作；
- \(T\)：版本化状态转移；
- \(O_i\)：Agent Observation；
- \(Z\)：信息过滤函数；
- \(U_i\)：Persona / Objective Utility；
- \(E\)：随机市场事件和公司事故；
- \(H\)：Episode Horizon。

Engineering MVP 使用：

\[
O_i=S
\]

作为完全信息 Baseline。

Research MVP 使用：

\[
O_i=Z_i(S),\quad O_i\neq S
\]

---

# 5. Engineering MVP 市场环境

市场模型的参数和完整算法不在本报告维护第二份副本。

唯一来源：

- `market_v4.yaml`：参数；
- Engineering MVP v4.0：算法、时序、RNG、Hash、幂等。

本报告只说明研究语义。

## 5.1 Company State

公司状态覆盖：

### Financial

- Cash；
- Revenue；
- Variable / Fixed / Incident Cost；
- Round / Cumulative Profit；
- Capacity Book Value。

### Commercial

- Price；
- Market Share；
- Potential Demand；
- Sales；
- Attempted Unfulfilled；
- Redistributed Orders。

### Operations

- Base / Effective Capacity 与版本化 Investment Pipeline；
- Unit Cost；
- Capacity Utilization；
- Financial Capacity。

### Brand

- Awareness；
- Service Quality；
- Reputation；
- Historical Stockout。

### Risk

- Resilience；
- Active Incident；
- Repair Progress。

这些字段让 Agent 的决策基于“公司处于什么经营状态”，而不是只看市场均价。

## 5.2 Market State

包括：

- Demand；
- Market Sentiment；
- Supply Cost Index；
- Consumer Segments；
- Outside Option；
- Risk Signals；
- Active Events；
- No-purchase / Lost Demand。

## 5.3 Numeric Action

最终可执行动作统一为：

```json
{
  "price_cents": 9450,
  "advertising_budget_cents": 1800000,
  "service_budget_cents": 1200000,
  "capacity_investment_cents": 1500000,
  "resilience_budget_cents": 600000,
  "incident_response": {
    "mode": "partial_repair",
    "repair_budget_cents": 1200000
  }
}
```

旧数据中的：

```text
price_level
advertising_level
capacity_level
service_investment
strategic_move
```

不再作为 Canonical Environment Action。

Preset 可以存在，但只作为 Numeric Action 的输入别名。

---

# 6. 市场决策 Trade-off

环境需要同时支持：

## Pricing

```text
Lower Price
→ 需求倾向增加
→ 单位贡献下降
→ 可能触发 Capacity / Cash 约束
```

## Advertising

```text
当前获客
+ Awareness 积累
- Cash
- 边际收益递减
```

## Service

```text
当前质量吸引力
+ 长期 Service / Reputation
- Cash
```

## Capacity

```text
当前投入
→ 下一轮生效
→ 未来履约能力
→ 有折旧和机会成本
```

## Resilience

```text
当前投入
→ 下一轮生效
→ 降低后续事件影响
→ 事件不发生时产生机会成本
```

## Repair

```text
Wait
vs
Partial Repair
vs
Full Repair
```

Agent 需要在维修支出和多轮事故损失之间权衡。

---

# 7. 消费者与市场分配

Research 和 Engineering 统一使用三类消费者：

| Segment | 主要偏好 |
|---|---|
| Price-sensitive | Relative Price |
| Quality-sensitive | Service / Reputation |
| Loyal | Reputation / Historical Fulfillment |

每个 Segment 的选择集包括：

```text
所有公司 + Outside Option
```

因此：

- 所有公司都差时，消费者可以不买；
- 市场成交量不再被强行等于 Base Demand；
- Stockout 后允许一次转售；
- 未转售成功的需求成为 Lost Demand。

这是后续 Consumer Welfare 和市场健康度评估的基础。

---

# 8. 现金、产能与终局

## 8.1 Cash

不允许：

- 负现金；
- 贷款；
- Engineering MVP 公司退出。

固定支出先校验。

当每单贡献为负时，使用 Financial Capacity 限制亏本销量，保证结算后 Cash 非负。

## 8.2 Capacity

- 投资下一轮生效；
- 有轻微折旧；
- Engineering MVP 无维护费和闲置费；
- Capacity Constraint 限制 Sales。

## 8.3 Terminal

Agent 能看到 `rounds_remaining`。

最后一轮禁止延迟生效的 Capacity / Resilience Investment。

Episode 结束计算 Terminal Enterprise Value，包含：

- Cash；
- Capacity Salvage；
- Awareness；
- Service；
- Reputation；
- Resilience。

---

# 9. 随机性、风险和主动响应

## 9.1 正常随机性

- Demand Noise；
- Market Sentiment；
- Supply Cost Noise；
- Consumer Utility Noise；
- Operational Capacity Noise。

## 9.2 重大 Market Event

- Extreme Weather；
- Supply Chain Shock；
- Regional Logistics Disruption；
- Festival Demand Surge。

所有重大负面事件至少提前一轮提供 Risk Signal。

## 9.3 Company Incident

- Platform System Outage；
- Warehouse Equipment Failure；
- Cold Chain Incident。

公司事故进入下一状态后，Agent 在其第一次影响销售前可以选择维修。

## 9.4 可复现性

必须满足：

```text
Same State
+ Same Joint Action
+ Same Seed
+ Same Environment Version
→ Same Next State Hash
```

RNG 与 Hash 使用 Engineering MVP v4 的固定协议。

---

# 10. Agent 架构

推荐：

```text
Fixed Round Workflow
        +
Dynamic Strategic Planner
```

单 Agent：

```text
Observation
↓
State Analysis
↓
Opponent / Risk Analysis
↓
Planner
↓
Candidate Numeric Actions
↓
Constraint-aware Selection
↓
Validator
↓
Final Action
↓
Outcome Analysis
↓
Memory Update
```

Engineering MVP 中：

- Planner 只需完成短期计划；
- 可看到完整信息；
- 不需要 Belief。

Research MVP 再加入：

- Opponent Belief；
- Multi-round Plan；
- Replanning；
- Game Theory Advisor。

---

# 11. Planner 输出

Planner 不输出不可更改的长期动作序列。

建议：

```json
{
  "objective": "preserve cash while preparing for demand surge",
  "horizon_rounds": 2,
  "market_assessment": "...",
  "company_constraints": ["capacity near limit"],
  "risk_assessment": "...",
  "candidate_actions": [
    {
      "price_cents": 9700,
      "advertising_budget_cents": 1000000,
      "service_budget_cents": 1200000,
      "capacity_investment_cents": 2000000,
      "resilience_budget_cents": 0,
      "repair_budget_cents": 0
    }
  ],
  "replan_triggers": []
}
```

最终 Executor 只提交当前 Round Action。

---

# 12. Persona

## 12.1 Engineering MVP

支持：

- None；
- Aggressive；
- Conservative；
- Balanced。

其 Utility 归一化和权重由 `market_v4.yaml` 定义。

## 12.2 Research MVP

加入：

- Cooperative。

Cooperative 只有在系统支持：

- Communication；
- Cooperation Action；
- Welfare / Trust；

后才有研究意义。

Environment 永远不根据 Persona 直接加成。

---

# 13. 完全信息与不完全信息

## 13.1 True State

最终 True State 包含：

- 完整 Market State；
- 所有 Company Financial / Commercial / Operations / Brand / Risk；
- Active / Pending Event；
- Risk Signals；
- Incident Repair；
- Investment Pipeline；
- RNG / Version Context。

## 13.2 Public State

Research MVP 可公开：

- Price；
- Market Share；
- Public Reputation；
- Public Service Score；
- Public Risk Signals；
- Active Market Event；
- 上一轮公开动作；
- 是否出现明显事故。

## 13.3 Private State

可以隐藏：

- Cash；
- Exact Unit Cost；
- Exact Capacity；
- Investment Pipeline；
- Repair Cost；
- Internal Utility；
- Long-term Plan。

## 13.4 Observation API

```python
observation = env.get_observation(agent_id, information_mode)
constraints = env.get_action_constraints(agent_id, state_version)
```

连续动作环境不再使用 `legal_actions()`。

---

# 14. Belief / Opponent Modeling

Research MVP 中，每个 Agent 维护：

```json
{
  "opponent_beliefs": {
    "company_B": {
      "estimated_cash_interval_cents": [],
      "estimated_capacity_interval": [],
      "price_response_distribution": {},
      "risk_preparation_probability_ppm": 0,
      "repair_strategy_distribution": {},
      "likely_persona": "conservative",
      "confidence_ppm": 0,
      "evidence_ids": []
    }
  }
}
```

Belief 必须区分：

- Observed Fact；
- Opponent Claim；
- Statistical Estimate；
- LLM Inference。

预测质量使用：

- Log Loss；
- Brier Score；
- Calibration；
- Numeric Action Error。

---

# 15. Communication、合作与竞争

## 15.1 Communication Phase

```text
Observe
→ Communicate / Negotiate
→ Close Communication
→ Private Decision
→ Action Lock
```

Communication 不能泄漏当前 Round Final Action。

## 15.2 Cheap Talk

非绑定消息：

- Cooperation Proposal；
- Competitive Signal；
- Threat；
- Stabilization Proposal；
- Risk Sharing Proposal。

## 15.3 结构化合作

Research MVP 可新增动作层，但不能破坏核心 Numeric Business Action。

例如：

```json
{
  "business_action": {},
  "cooperation_action": {
    "type": "shared_logistics",
    "counterparty": "company_C",
    "offered_budget_cents": 500000,
    "accept_proposal_id": "proposal-17"
  }
}
```

合作只有双方接受才生效。

## 15.4 Credibility

Credibility 由历史履约计算，不由发送者自己声明。

---

# 16. 社会福利

Research MVP 统一在新市场上计算。

## 16.1 Consumer Welfare

至少包含：

- 成交消费者 Utility；
- Payment；
- Outside-option Utility；
- Attempted Stockout Cost；
- Lost Demand Cost；
- Service / Reputation Benefit。

## 16.2 Producer Welfare

\[
PW=\sum_i Profit_i
\]

以及：

- Cash Stability；
- Firm Survival proxy；
- Profit Variance。

## 16.3 Social Welfare Proxy

\[
SW=
CW+PW
-Waste
-ServiceFailure
-MarketConcentrationPenalty
-ExcessiveAdvertisingCost
-DisasterExternality
\]

必须明确这是项目定义的实验 Proxy，不是唯一经济学标准。

## 16.4 HHI

基于实际成交 Market Share，而非初始 Potential Demand Share。

---

# 17. Game Theory Advisor

Advisor 不能只输出文本“使用博弈论”。

连续动作环境下输出：

```json
{
  "candidate_numeric_actions": [],
  "predicted_opponent_action_distributions": {},
  "scenario_rollouts": [
    {
      "scenario": "competitors cut price",
      "probability_ppm": 450000,
      "predicted_profit_cents": 0,
      "predicted_terminal_value_cents": 0,
      "price_war_risk_ppm": 0
    }
  ],
  "short_term_recommendation": {},
  "long_term_recommendation": {},
  "deviation_risk": {},
  "confidence_ppm": 0
}
```

Game Theory Advisor 必须读取：

- Action Constraints；
- Remaining Rounds；
- Risk Signals；
- Company Liquidity；
- Opponent Beliefs。

---

# 18. Search 与 Approximate Best Response

连续动作不能枚举全部合法动作。

推荐流程：

```text
LLM / Rule Proposal
→ 生成有限 Candidate Numeric Actions
→ Simulator Rollout
→ Value / Utility Evaluation
→ 选择 Action
```

Candidate 来源：

- Planner；
- Local Numeric Perturbation；
- Preset Anchor；
- Historical Successful Policy；
- Random Feasible Candidate。

近似 Best Response：

\[
\widehat{BR}_i(\sigma_{-i})
=
\arg\max_{a\in C_i}
\widehat{E}[U_i|a,\sigma_{-i}]
\]

其中 \(C_i\) 是有限候选集。

报告必须称为 Approximate Best Response，不能声称找到连续空间严格最优解。

---

# 19. Self-play 与 PSRO

## 19.1 Policy

Policy 是可重复执行的决策程序：

```yaml
policy_id: cautious-risk-aware-v3
planner_version: planner-v4
persona: conservative
candidate_generator: numeric-local-search-v1
risk_threshold_ppm: 650000
repair_threshold_ppm: 400000
```

不是单个动作。

## 19.2 Policy Population

包含：

- Aggressive Growth；
- Margin Protection；
- Service / Reputation；
- Risk-aware；
- Conditional Retaliator；
- Cooperative（Research 阶段）；
- Game-theory-informed。

## 19.3 Empirical Payoff

使用相同 `market_v4.yaml`、环境版本和成组 Seed 评估 Policy 组合。

四人市场较复杂，推荐：

- 先在两公司简化配置上做 PSRO；
- 主四公司市场做 Focal-policy vs Opponent-mixture。

## 19.4 DCH

DCH 作为 Population 规模扩大后的 Stretch Goal，不属于 Engineering / Research MVP 必须实现。

---

# 20. Baselines

统一 Baseline：

1. Random Feasible Numeric Agent；
2. Rule Agent；
3. Persona-only LLM；
4. LLM + Memory；
5. LLM + Belief；
6. LLM + Belief + Game Theory Advisor；
7. Search / Rollout Agent；
8. PSRO Population Policy。

所有 Baseline 使用同一后端 MarketEnv。

---

# 21. Evaluation

## 21.1 工程可靠性

- Episode Completion；
- Raw Action Invalid；
- Retry / Fallback；
- Final Illegal Execution；
- State Invariant；
- Idempotency Conflict；
- Replay Match；
- Latency / Tokens / Cost。

## 21.2 公司战略表现

- Round / Cumulative Profit；
- Cash；
- Market Share；
- Terminal Enterprise Value；
- Capacity Utilization；
- Stockout；
- Awareness / Service / Reputation；
- Resilience；
- Disaster Loss；
- Incident Repair Cost / Duration。

## 21.3 Persona

- Numeric Action Distribution；
- Risk Preparation；
- Repair Strategy；
- Profit Volatility；
- Share Growth；
- Utility Consistency。

## 21.4 信息与 Belief

- Opponent Numeric Action Prediction；
- Hidden State Estimate Error；
- Calibration；
- Decision Utility。

## 21.5 Game Theory

- Estimated Deviation Gain；
- Approximate Regret；
- Cross-play Robustness；
- Price-war Frequency；
- Cooperation Stability。

## 21.6 Welfare

- Consumer Welfare；
- Producer Welfare；
- Social Welfare Proxy；
- HHI；
- No-purchase；
- Lost Demand；
- Service Failure。

---

# 22. 核心实验

## Experiment 0：Environment Validity

验证动作影响大于普通噪声，并完成 RNG / Replay / Incident / Outside-option 测试。

## Experiment 1：Persona Effect

固定 State、Model、Seed，只改变 Persona。

## Experiment 2：Information Structure

Perfect vs Imperfect vs Noisy。

## Experiment 3：Communication

Off vs Cheap Talk vs Structured Cooperation。

## Experiment 4：Risk / Disaster

有预警时，不同 Persona 是否进行不同 Resilience / Capacity 准备。

## Experiment 5：Incident Response

Wait vs Partial vs Full Repair。

## Experiment 6：Game Theory Advisor

Belief-only vs Belief + Advisor。

## Experiment 7：Welfare-aware

Company Utility vs Company + Welfare Constraint。

## Experiment 8：Cross-play

已见与未见 Policy / Persona / Market Regime。

## Experiment 9：Population

Homogeneous Self-play vs Mixed Population vs PSRO-inspired。

## Experiment 10：Dataset Policy

使用交互数据训练小型 Numeric Policy，并在 Holdout 环境测试。

所有对照优先使用配对 Seed。

---

# 23. 数据体系

旧离散动作数据结构全部废弃。

## 23.1 EpisodeManifest

保存：

- Config / Environment / RNG / Hash / Schema Version；
- Seed；
- Code Commit；
- Model / Prompt / Persona；
- Initial State Snapshot；
- Initial State Hash。

## 23.2 Round Transition

```json
{
  "episode_id": "...",
  "round": 5,
  "state_before": {},
  "observations": {},
  "action_constraints": {},
  "raw_actions": {},
  "validation_results": {},
  "final_numeric_actions": {},
  "joint_action": {},
  "random_draw_summary": {},
  "market_events": {},
  "company_incidents": {},
  "company_outcomes": {},
  "state_after": {},
  "done": false
}
```

## 23.3 Agent Decision

```json
{
  "agent_id": "company_A",
  "persona": "aggressive",
  "observation_ref": "...",
  "constraint_ref": "...",
  "planner_summary": {},
  "raw_action": {},
  "final_action": {
    "price_cents": 9450,
    "advertising_budget_cents": 1800000,
    "service_budget_cents": 1200000,
    "capacity_investment_cents": 1500000,
    "resilience_budget_cents": 600000,
    "incident_response": {
      "mode": "partial_repair",
      "repair_budget_cents": 1200000
    }
  },
  "utility_components": {},
  "validation": {}
}
```

## 23.4 RL / Policy Transition

```json
{
  "observation": {},
  "action_constraints": {},
  "action": {},
  "environment_outcomes": {},
  "agent_utility": {},
  "next_observation": {},
  "event_context": {},
  "done": false
}
```

## 23.5 Communication / Belief

在对应 Research Stage 才新增，不污染 Engineering MVP Schema。

---

# 24. Counterfactual 数据

在完全相同：

- State；
- Seed；
- Opponent Actions；

下改变：

- Persona；
- Advisor On / Off；
- Risk Preparation；
- Repair Action；
- Communication Message。

记录 Numeric Action 和 Outcome 差异。

Counterfactual 必须引用相同 `state_before_hash`。

---

# 25. 软件架构

```text
project/
├── configs/
│   └── market_v4.yaml
├── environment/
│   ├── market_env.py
│   ├── state.py
│   ├── consumer_choice.py
│   ├── allocation.py
│   ├── liquidity.py
│   ├── events.py
│   ├── incidents.py
│   ├── rng_protocol.py
│   ├── canonical_json.py
│   └── invariants.py
├── agents/
│   ├── base.py
│   ├── rule.py
│   ├── mock.py
│   ├── llm.py
│   ├── planner.py
│   ├── belief.py
│   └── game_theory_advisor.py
├── controller/
│   ├── episode_runner.py
│   ├── action_lock.py
│   └── idempotency.py
├── validation/
│   ├── action_schema.py
│   ├── constraints.py
│   └── fallback.py
├── data/
│   ├── manifest.py
│   ├── event_log.py
│   ├── replay.py
│   └── dataset_export.py
├── evaluation/
│   ├── engineering.py
│   ├── strategy.py
│   ├── welfare.py
│   ├── belief.py
│   └── game_theory.py
├── frontend/
│   └── display_only/
└── tests/
```

---

# 26. 核心接口

## Environment

```python
class MarketEnv:
    def reset(self, manifest): ...
    def get_state(self): ...
    def get_observation(self, agent_id, information_mode): ...
    def get_action_constraints(self, agent_id, state_version): ...
    def validate_action(self, action): ...
    def step(self, step_id, joint_action): ...
    def replay(self, manifest, joint_actions): ...
```

## Agent

```python
class Agent:
    def observe(self, observation, constraints): ...
    def plan(self): ...
    def act(self): ...
    def analyze_result(self, step_result): ...
```

## Policy

```python
class Policy:
    def select_numeric_action(self, observation, constraints): ...
```

---

# 27. Controller Workflow

```python
state = env.get_state()

observations = {
    agent_id: env.get_observation(agent_id, information_mode)
    for agent_id in agents
}

constraints = {
    agent_id: env.get_action_constraints(agent_id, state.state_version)
    for agent_id in agents
}

raw_actions = run_agents_from_same_state(
    observations,
    constraints
)

final_actions = validator.validate_or_fallback_all(
    raw_actions,
    constraints
)

joint_action = action_lock(final_actions)

step_result = env.step(
    step_id=f"{episode_id}:{state.round}:{state.state_version}",
    joint_action=joint_action
)

logger.append(step_result)
evaluator.update(step_result)
```

---

# 28. 测试与 Benchmark

## Engineering Benchmark

- Config；
- Formula；
- Time Order；
- Outside Option；
- Redistribution；
- Cash；
- Terminal；
- Event Lifecycle；
- Incident Repair；
- RNG Vectors；
- Canonical JSON / Hash；
- Idempotency；
- Replay。

## Agent Benchmark

- Valid Numeric Action；
- Constraint Compliance；
- Risk Response；
- Incident Response；
- Persona Consistency；
- Stability / Cost。

## Research Benchmark

- Information；
- Belief；
- Communication；
- Advisor；
- Welfare；
- Cross-play；
- Generalization。

---

# 29. 开发路线

## Phase 0：规范冻结

- `market_v4.yaml`；
- State / Action Schema；
- RNG / Hash Test Vectors；
- EpisodeManifest。

## Phase 1：Engineering MVP

- Backend MarketEnv；
- Rule / Mock；
- Controller；
- Replay；
- Frontend display-only。

## Phase 2：Basic LLM / Persona

- Numeric Action Prompt；
- Constraint-aware output；
- Persona paired tests。

## Phase 3：Imperfect Information

- Observation Filter；
- Belief；
- Prediction Evaluation。

## Phase 4：Communication / Cooperation / Welfare

- Cheap Talk；
- Structured Proposal；
- Cooperative Persona；
- Welfare。

## Phase 5：Game Theory

- Advisor；
- Numeric Candidate Search；
- Approximate Regret。

## Phase 6：Self-play / Cross-play

- Opponent Pool；
- Environment Shift；
- Population。

## Phase 7：PSRO / Dataset Training

- Empirical Game；
- Meta-strategy；
- Approximate BR；
- Policy Model。

---

# 30. 主要风险

## 市场公式决定一切

缓解：

- Sensitivity；
- Parameter Shift；
- Config Version；
- 不同 Regime。

## 随机性压倒动作

缓解：

- Agency vs Noise Test；
- 配对 Seed；
- Effect Size。

## LLM 解释好看但策略无效

缓解：

- Numeric Outcomes；
- Baselines；
- Counterfactual；
- Approximate Regret。

## 数据不可复现

缓解：

- Manifest；
- Stable RNG；
- JCS Hash；
- Joint Action Replay。

## 前后端漂移

缓解：

- Backend-only MarketEnv；
- Frontend display-only；
- State Hash 验证。

---

# 31. 最终 Demo

展示一个完整 Round：

```text
Market State / Risk Signal
↓
Company State
↓
Observation + Action Constraints
↓
Planner Candidate Numeric Actions
↓
Raw / Final Action
↓
Action Lock
↓
Event / Incident / Random Draw Summary
↓
Consumer Allocation + Outside Option
↓
Capacity / Cash Constraint
↓
Profit / Cash / Reputation / Resilience
↓
Next State Hash
```

再展示：

- Persona 对比；
- Risk Preparation；
- Incident Repair；
- Perfect / Imperfect；
- Advisor On / Off；
- Cross-play；
- Welfare。

---

# 32. 最终成功标准

项目最终成功不是“多个 Agent 可以聊天”，而是能够用可复现数据回答：

> 在动态随机生鲜市场中，不同 Persona、信息结构、沟通机制和博弈论推理，是否会改变 LLM Agent 的连续资源配置决策？这些改变是否提高公司长期价值、对未知对手的稳健性或社会福利？交互数据能否进一步训练出更便宜、更稳定的策略模型？

即使结果发现某个 LLM 或 Advisor 没有改善，也属于有效研究结论，只要：

- 环境正确；
- 对照严谨；
- 数据可复现；
- 指标可解释。

---

# 33. 参考方向

- 项目原始题目：LLM-Guided Game-Theoretical Modeling for Strategic Decision-Making；
- 多智能体重复博弈；
- Imperfect-information games；
- OpenSpiel / RLCard；
- MCTS / CFR；
- Policy Space Response Oracles；
- Deep Cognitive Hierarchies；
- LLM Agent Planning / Opponent Modeling。

---

# 34. 一句话定义

> 构建一个以后端为唯一计算源、具有连续资源配置、现金与产能约束、消费者外部选项、风险预警、主动事故维修和版本化可复现随机性的多 Agent 生鲜市场，并在其上研究 Persona、不完全信息、合作、福利、博弈论推理、自博弈和策略训练。
