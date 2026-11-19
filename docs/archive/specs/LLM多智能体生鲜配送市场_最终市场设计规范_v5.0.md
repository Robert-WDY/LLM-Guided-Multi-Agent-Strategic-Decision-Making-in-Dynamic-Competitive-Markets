# LLM 多智能体生鲜配送市场
## 最终市场设计规范 v5.0

---

## 0. 文档定位

本文档定义项目最终希望达到的市场形态。

它不是单纯的“调价格、算利润”环境，而是一个同时包含以下机制的动态重复博弈市场：

- 公司内部经营状态；
- 多公司竞争；
- 消费者分群选择；
- 跨回合品牌、服务、声誉与产能积累；
- 可复现随机性；
- 灾难预警与主动风险准备；
- 公司事故与主动维修；
- 个人公司利益；
- 消费者利益；
- 公共品与社会福利；
- 合作、搭便车和背叛；
- 完全信息与不完全信息；
- Communication、Commitment 和 Signaling；
- Game-Theoretic Advisor；
- Self-play、Cross-play 和策略 Population。

最终市场希望真实反映以下现象：

```text
所有公司共同贡献
→ 公共基础设施、消费者信任和行业韧性提高
→ 社会福利提高
→ 所有公司长期经营环境改善

某家公司选择少贡献或不贡献
→ 节省当期成本
→ 短期私人收益可能高于贡献者
→ 同时继续享受其他公司的公共贡献
→ 形成搭便车优势

越来越多公司选择自私和搭便车
→ 公共品存量下降
→ 消费者信任下降
→ 事故和灾难损失上升
→ 供应成本上升
→ 市场需求下降
→ 社会福利进一步下降
→ 最终所有公司的长期利益也受损
```

该设计研究的不是：

> LLM 会不会选择看起来“善良”的答案。

而是：

> LLM 能否在私人利益、集体利益、短期诱惑、长期收益、风险、对手行为和不完全信息之间做出真正的战略选择。

---

# 1. 核心设计原则

## 1.1 Environment 不读取 Persona

禁止：

```python
if persona == "cooperative":
    social_welfare += bonus
```

正确流程：

```text
Persona / Utility / Belief
→ Agent Planner
→ Agent Action
→ 统一 MarketEnv
→ 经营结果与社会结果
```

所有 Agent 面对完全相同的市场公式。

人格只能通过动作影响市场。

---

## 1.2 公司利益与社会利益分开计算

必须区分：

### Company Outcome

- Revenue；
- Profit；
- Cash；
- Market Share；
- Enterprise Value；
- Risk Loss；
- Reputation。

### Consumer Outcome

- 消费者效用；
- 实际支付；
- 服务质量；
- 缺货；
- Outside Option；
- Variety。

### Public / Social Outcome

- 公共品存量；
- 消费者信任；
- 市场韧性；
- 外部性；
- 市场集中度；
- 社会福利。

公司利润增加不代表社会福利一定增加。

---

## 1.3 私人短期最优不一定等于长期最优

公共贡献必须满足：

```text
单家公司贡献的即时私人回报
<
其贡献成本

但

所有公司获得的总社会回报
>
其贡献成本
```

因此搭便车短期有吸引力。

同时公共品下降必须逐步损害：

- Demand；
- Supply Cost；
- Event Loss；
- Incident Probability；
- Consumer Trust。

这样，自私但长期理性的 Agent 仍可能选择合作。

---

## 1.4 随机性不能压倒 Agency

普通噪声用于制造不确定性。

但必须满足：

> 关键动作的平均影响明显大于普通市场噪声。

灾难可以是低概率、高影响事件，但：

- 重大灾难必须提前提供 Risk Signal；
- Agent 必须有准备和响应手段；
- 公司事故必须允许 Wait、Partial Repair、Full Repair。

---

## 1.5 同输入必须可重放

必须满足：

```text
相同 State
+
相同 Joint Action
+
相同 Episode Seed
+
相同 Environment Version
+
相同 Config Version
=
相同 Next State
```

Seed 不提供给 Agent。

Seed 只用于：

- Manifest；
- Debug；
- Replay；
- Counterfactual。

---

## 1.6 市场只有一个计算源

唯一权威：

```text
Backend MarketEnv.step()
```

禁止：

- 前端自行计算利润；
- Agent 直接修改状态；
- Evaluator 复制一套市场公式；
- 不同实验使用未版本化公式。

---

# 2. 市场整体分层

最终市场分为五层。

```text
┌─────────────────────────────────────┐
│ 1. Company Competition Layer        │
│ Price / Ad / Service / Capacity     │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 2. Consumer Choice Layer            │
│ Price / Quality / Loyalty / Outside │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 3. Public-Goods & Social Layer      │
│ Contribution / Trust / Resilience   │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 4. Risk & Event Layer               │
│ Warning / Disaster / Incident       │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ 5. Information & Game Layer         │
│ Communication / Belief / Advisor    │
└─────────────────────────────────────┘
```

---

# 3. 核心市场闭环

```text
state_t
│
├── Market State
├── Public-Good State
├── Company States
├── Consumer State
├── Risk Signals
├── Active Events
├── Company Incidents
├── Communication History
└── Current Plans / Commitments
        ↓
Agent-specific Observation
        ↓
Communication / Commitment
        ↓
Private Planning
        ↓
Commercial Action
+
Risk Action
+
Social Contribution
+
Incident Response
        ↓
Action Validation
        ↓
Action Lock
        ↓
Joint Action
        ↓
MarketEnv.step()
        ↓
Consumer Choice
        ↓
Capacity and Financial Constraint
        ↓
Sales / Profit / Stockout
        ↓
Public-Good Update
        ↓
Trust / Reputation / Risk Update
        ↓
Social Welfare
        ↓
state_t+1
```

---

# 4. MarketState

推荐结构：

```json
{
  "meta": {
    "episode_id": "episode-001",
    "round": 8,
    "max_rounds": 30,
    "rounds_remaining": 22,
    "state_version": 8,
    "environment_version": "market-v5.0.0",
    "config_version": "market-v5-default"
  },

  "demand": {
    "base_demand_orders": 10000,
    "realized_demand_orders": 10840,
    "outside_option_orders": 920
  },

  "macro": {
    "supply_cost_index_ppm": 1080000,
    "market_sentiment_ppm": 1020000,
    "average_price_cents": 9800,
    "hhi_ppm": 268000
  },

  "public_goods": {},

  "consumers": {},

  "risk_signals": [],

  "active_market_events": [],

  "public_projects": [],

  "companies": {},

  "last_joint_action": {},

  "state_hash": "sha256:..."
}
```

---

# 5. CompanyState

每家公司必须维护能够驱动真实决策的内部状态。

```json
{
  "company_id": "company_A",

  "identity": {
    "persona_id": "selfish-long-term",
    "policy_version": "llm-agent-v3"
  },

  "financial": {
    "cash_balance_cents": 25000000,
    "round_revenue_cents": 21000000,
    "round_variable_cost_cents": 13000000,
    "round_fixed_spend_cents": 4200000,
    "round_incident_cost_cents": 400000,
    "round_profit_cents": 3400000,
    "cumulative_profit_cents": 18000000,
    "enterprise_value_cents": 42000000
  },

  "commercial": {
    "price_cents": 9600,
    "market_share_ppm": 270000,
    "potential_demand_orders": 3200,
    "sales_orders": 2800,
    "unfulfilled_demand_orders": 400,
    "orders_received_from_redistribution": 160,
    "orders_lost_after_redistribution": 240
  },

  "operations": {
    "base_capacity_orders": 3000,
    "effective_capacity_orders": 2800,
    "pending_capacity_orders": 500,
    "capacity_utilization_ppm": 1000000,
    "unit_cost_cents": 5800
  },

  "brand": {
    "brand_awareness_ppm": 540000,
    "service_quality_ppm": 720000,
    "reputation_ppm": 680000
  },

  "risk": {
    "resilience_ppm": 350000,
    "pending_resilience_ppm": 80000,
    "active_incident": null
  },

  "social": {
    "last_public_contribution_cents": 200000,
    "cumulative_public_contribution_cents": 1800000,
    "cooperation_reputation_ppm": 640000,
    "commitment_credibility_ppm": 580000,
    "free_rider_score_ppm": 310000
  },

  "history": {
    "last_action": {},
    "recent_profit_cents": [],
    "recent_market_share_ppm": []
  }
}
```

---

# 6. PublicGoodState

公共品不能只表示为一个抽象的 Social Welfare 数字。

推荐维护以下真实公共状态。

```json
{
  "shared_infrastructure_ppm": 520000,
  "consumer_trust_ppm": 680000,
  "industry_resilience_ppm": 610000,
  "regulatory_pressure_ppm": 180000,
  "environmental_externality_ppm": 270000,

  "last_total_contribution_cents": 5000000,
  "contribution_history_cents": [],

  "collective_health_regime": "stable"
}
```

---

## 6.1 Shared Infrastructure

表示：

- 共享冷链；
- 公共物流；
- 行业安全系统；
- 事故应急资源。

提高后可以：

- 降低供应成本；
- 减少 Company Incident；
- 降低灾难严重度；
- 提高履约能力。

---

## 6.2 Consumer Trust

表示消费者对整个行业的信任。

受以下因素影响：

- 平均服务；
- 平均履约；
- 大规模事故；
- 虚假承诺；
- 严重价格操纵；
- 公共贡献。

提高后：

- Outside Option 下降；
- 整体 Demand 上升。

下降后：

- 更多消费者选择不购买；
- 所有公司都受到影响。

---

## 6.3 Industry Resilience

表示整个市场对灾害的共同防御能力。

它与单家公司 `resilience` 不同：

```text
Company Resilience
→ 保护单家公司

Industry Resilience
→ 降低全市场事件损失
```

---

## 6.4 Regulatory Pressure

受以下因素提高：

- 高市场集中度；
- 联合抬价；
- 长期消费者福利下降；
- 虚假承诺；
- 严重服务事故。

提高后可以产生：

- 罚款；
- 广告限制；
- 价格监管；
- 强制公共贡献。

---

## 6.5 Environmental Externality

用于衡量：

- 过度运力；
- 库存浪费；
- 配送资源浪费；
- 过度广告；
- 高碳运输。

该指标主要进入社会福利。

---

# 7. Consumer Segments

最终市场至少包含三类消费者。

| Segment | 默认占比 | 核心偏好 |
|---|---:|---|
| Price-sensitive | 45% | 价格 |
| Quality-sensitive | 35% | 服务与履约 |
| Loyal / Brand-sensitive | 20% | 声誉、信任和历史体验 |

消费者还拥有：

> Outside Option：不购买或离开当前市场。

---

# 8. Agent Action

最终 Canonical Action：

```json
{
  "action_id": "episode-001-r08-company-A-v1",
  "episode_id": "episode-001",
  "agent_id": "company_A",
  "round": 8,
  "state_version": 8,

  "commercial_action": {
    "price_cents": 9450,
    "advertising_budget_cents": 1800000,
    "service_budget_cents": 1200000,
    "capacity_investment_cents": 1500000
  },

  "risk_action": {
    "resilience_budget_cents": 600000,
    "incident_response": {
      "mode": "partial_repair",
      "repair_budget_cents": 1200000
    }
  },

  "social_action": {
    "public_contribution_cents": 500000,
    "project_contributions": {
      "shared_cold_chain_project": 300000
    }
  },

  "commitment_references": [
    "commitment-017"
  ]
}
```

---

# 9. Action 输入模式

支持：

## Numeric Action

LLM 直接给具体数值。

## Preset Action

Rule Agent 或 Demo 使用：

```text
low
medium
high
```

Preset 必须转换成 Numeric Action 后再进入环境。

环境不保存 `"high"` 作为真实经营状态。

---

# 10. 回合精确时序

必须统一为：

\[
state_t + joint\_action_t + random_t
\rightarrow
result_t + state_{t+1}
\]

---

## 10.1 state_t 中已经确定的信息

Agent 决策前可以看到：

- 当前公司状态；
- 当前市场状态；
- 当前 Active Market Event；
- 当前公司 Incident；
- 当前 Public-Good State；
- 对未来一至两轮的 Risk Signal；
- 当前合法动作约束；
- Communication History；
- 当前 Plan。

---

## 10.2 Communication Phase

```text
Observation
→ Public / Private Communication
→ Commitment
→ Communication Close
```

Communication Close 后：

- 不允许继续修改消息；
- Agent 不知道对手当前回合最终动作；
- 所有公司进入私人决策。

---

## 10.3 Action Lock

所有 Agent 基于同一个不可变 `state_t` 生成动作。

```text
Raw Actions
→ Validator
→ Final Actions
→ Action Lock
→ Joint Action
```

每个 Round 只调用一次：

```python
MarketEnv.step(joint_action)
```

---

## 10.4 本轮立即生效

以下动作在当轮市场结算前生效：

- Price；
- Advertising；
- Service；
- Incident Repair；
- Public Contribution 的现金支出。

其中：

### Repair

在当轮销售前生效。

Partial Repair 会降低本轮事故影响。

Full Repair 可以在本轮市场结算前解除主要事故效果。

---

## 10.5 下一轮生效

以下动作从 `state_t+1` 生效：

- Capacity Investment；
- Resilience Investment；
- Public-Good Stock 增量；
- Threshold Project 成功后的公共效果。

所以：

```text
Round t 看见 Round t+1 灾难预警
→ Round t 投入 Resilience
→ Round t+1 可以发挥保护作用
```

但：

```text
Round t 已经发生灾难
→ Round t 才投入 Resilience
→ 不能抵消已经激活的灾难
```

---

## 10.6 当前 Active Event

`state_t` 中的 Active Event 影响当前 Round。

多个事件同时发生时：

- Cost Multiplier 使用乘法；
- Capacity Multiplier 使用乘法；
- Demand Multiplier 使用乘法；
- Service / Reputation Penalty 使用加法后 Clip；
- 任何字段都必须经过上下界限制。

---

## 10.7 当前公司 Incident

当前 Incident 在维修后影响当轮：

```text
Incident Raw Severity
×
Remaining Repair Severity
×
Company Resilience Modifier
```

---

## 10.8 市场结算

```text
Normal Random Draw
→ Consumer Utility
→ First-choice Demand
→ Capacity / Financial Constraint
→ Sales
→ One-time Redistribution
→ Revenue / Cost / Profit
```

---

## 10.9 长期状态更新

结算后更新：

- Awareness；
- Service Quality；
- Reputation；
- Cash；
- Capacity；
- Resilience；
- Public-Good Stock；
- Consumer Trust；
- Regulatory Pressure；
- Environmental Externality。

---

## 10.10 Event Lifecycle

Round 结束后：

1. 当前 Active Event 剩余时长减一；
2. 已到期事件结束；
3. 目标 Round 为 `t+1` 的 Risk Signal 进行实现抽样；
4. 实现的事件进入 `state_t+1`；
5. 生成新的未来 Risk Signal；
6. 为每家公司采样下一轮 Company Incident；
7. 生成 `state_t+1`。

---

# 11. 价格与获客机制

## 11.1 Relative Price Signal

\[
PriceSignal_i
=
\frac{\bar P-P_i}{PriceScale}
\]

价格低于市场平均：

> 正值。

---

## 11.2 Advertising Input

\[
AdInput_i
=
1-\exp(-AdBudget_i/AdScale)
\]

广告边际收益递减。

---

## 11.3 Service Input

\[
ServiceInput_i
=
1-\exp(-ServiceBudget_i/ServiceScale)
\]

---

## 11.4 Effective Awareness

\[
EffectiveAwareness_i
=
clip(
0.7Awareness_{i,t}
+
0.3AdInput_i,
0,
1
)
\]

---

## 11.5 Effective Service

\[
EffectiveService_i
=
clip(
0.7ServiceQuality_{i,t}
+
0.3ServiceInput_i,
0,
1
)
\]

---

# 12. 消费者效用

以下为示意规范，最终系数必须统一放入版本化配置。

## 12.1 Price-sensitive

\[
V_{i,p}
=
1.4PriceSignal_i
+
0.45EffectiveAwareness_i
+
0.25EffectiveService_i
+
0.20Reputation_i
+
0.20ConsumerTrust_t
-
0.60LastStockoutRate_i
+
\epsilon_{i,p}
\]

---

## 12.2 Quality-sensitive

\[
V_{i,q}
=
0.50PriceSignal_i
+
0.25EffectiveAwareness_i
+
1.10EffectiveService_i
+
0.60Reputation_i
+
0.35ConsumerTrust_t
-
0.80LastStockoutRate_i
+
\epsilon_{i,q}
\]

---

## 12.3 Loyal

\[
V_{i,l}
=
0.35PriceSignal_i
+
0.15EffectiveAwareness_i
+
0.50EffectiveService_i
+
1.20Reputation_i
+
0.50ConsumerTrust_t
-
1.00LastStockoutRate_i
+
\epsilon_{i,l}
\]

---

# 13. Outside Option

每个 Segment 都包含一个 Outside Option。

例如：

\[
V_{outside,s}
=
BaseOutsideUtility_s
-
\beta_T ConsumerTrust_t
+
\beta_P IndustryPricePressure_t
+
\beta_F IndustryFailureRate_t
\]

然后：

\[
Share_{i,s}
=
\frac{\exp(V_{i,s}/\tau)}
{
\exp(V_{outside,s}/\tau)
+
\sum_j\exp(V_{j,s}/\tau)
}
\]

因此：

- 所有公司价格太高；
- 服务都很差；
- 行业信任下降；

都会导致消费者不购买。

---

# 14. Potential Demand

\[
PotentialDemand_i
=
\sum_s Demand_s Share_{i,s}
\]

---

# 15. Capacity 与财务约束

## 15.1 Physical Capacity

\[
PhysicalCapacity_i
=
BaseCapacity_i
\times
EventCapacityMultiplier_i
\times
IncidentCapacityMultiplier_i
\times
OperationalNoise_i
\]

---

## 15.2 亏损订单的现金约束

每单净贡献：

\[
MarginPerOrder_i
=
Price_i
-
ActualUnitCost_i
-
IncidentRefundCostPerOrder_i
\]

如果：

\[
MarginPerOrder_i \ge 0
\]

则销售不会因净亏损消耗现金。

如果：

\[
MarginPerOrder_i < 0
\]

允许的亏损订单上限：

\[
FinancialCapacity_i
=
\left\lfloor
\frac{CashAfterFixedSpend_i}
{|MarginPerOrder_i|}
\right\rfloor
\]

最终：

\[
EffectiveCapacity_i
=
\min(
PhysicalCapacity_i,
FinancialCapacity_i
)
\]

这保证现金不会无解释地变为负数。

---

# 16. Sales 与缺货

\[
FirstPassSales_i
=
\min(
PotentialDemand_i,
EffectiveCapacity_i
)
\]

\[
UnfulfilledDemand_i
=
\max(
0,
PotentialDemand_i-FirstPassSales_i
)
\]

---

# 17. 缺货后二次分配

MVP 和最终市场均采用：

> 一次确定性 Redistribution，不递归。

未满足需求在仍有剩余 Capacity 的公司之间重新分配。

重新分配仍根据：

- Consumer Utility；
- Remaining Capacity；
- Outside Option；

计算。

第二次仍无法满足的订单记为：

```text
orders_lost_after_redistribution
```

---

# 18. 整数订单分配

所有订单最终必须为整数。

使用：

> Largest Remainder Method。

步骤：

1. 计算浮点期望订单；
2. 取 floor；
3. 按小数余数从大到小分配剩余订单；
4. 相同余数时按 `company_id` 排序。

这样可以保证跨语言一致。

---

# 19. 财务计算

## Revenue

\[
Revenue_i
=
Price_i \times Sales_i
\]

## Actual Unit Cost

\[
ActualUnitCost_i
=
BaseUnitCost_i
\times
SupplyCostIndex_t
\times
PublicInfrastructureCostMultiplier_t
\]

## Variable Cost

\[
VariableCost_i
=
ActualUnitCost_i
\times
Sales_i
\]

## Decision Spend

\[
DecisionSpend_i
=
AdBudget_i
+
ServiceBudget_i
+
CapacityInvestment_i
+
ResilienceBudget_i
+
RepairBudget_i
+
PublicContribution_i
+
ProjectContribution_i
\]

## Profit

\[
Profit_i
=
Revenue_i
-
VariableCost_i
-
DecisionSpend_i
-
IncidentCost_i
-
RegulatoryFine_i
\]

## Cash

\[
Cash_{i,t+1}
=
Cash_{i,t}
+
Profit_i
\]

通过动作校验与 Financial Capacity 保证：

\[
Cash_{i,t+1} \ge 0
\]

---

# 20. Capacity 更新

\[
CapacityAddition_i
=
CapacityInvestment_i/CapacityUnitCost
\]

\[
BaseCapacity_{i,t+1}
=
(1-DepreciationRate)
BaseCapacity_{i,t}
+
PendingCapacityAddition_{i,t}
\]

\[
PendingCapacityAddition_{i,t+1}
=
CapacityAddition_i
\]

---

# 21. Awareness、Service 与 Reputation

## Awareness

\[
Awareness_{i,t+1}
=
clip(
\rho_A Awareness_{i,t}
+
\eta_A AdInput_i,
0,
1
)
\]

## Service

\[
ServiceQuality_{i,t+1}
=
clip(
\rho_S ServiceQuality_{i,t}
+
\eta_S ServiceInput_i
-
EventServicePenalty_i
-
IncidentServicePenalty_i,
0,
1
)
\]

## Reputation

\[
Reputation_{i,t+1}
=
clip(
\rho_R Reputation_{i,t}
+
\eta_Q EffectiveService_i
+
\eta_F FulfillmentRate_i
+
\eta_H CommitmentHonesty_i
-
IncidentReputationPenalty_i
-
StockoutPenalty_i,
0,
1
)
\]

---

# 22. Resilience

## Company Resilience

\[
CompanyResilience_{i,t+1}
=
clip(
\rho_{CR}CompanyResilience_{i,t}
+
\eta_{CR}
(1-\exp(-ResilienceBudget_i/Scale)),
0,
1
)
\]

## Event Impact

\[
EffectiveEventImpact_i
=
RawEventImpact
\times
(1-\beta_C CompanyResilience_i)
\times
(1-\beta_I IndustryResilience_t)
\]

---

# 23. Market Randomness

随机性分为三类。

## 23.1 Normal Noise

- Demand Noise；
- Consumer Preference Noise；
- Operational Noise。

普通噪声必须小于主要 Action Effect。

## 23.2 Market Event

- Extreme Weather；
- Supply Chain Shock；
- Regional Logistics Disruption；
- Festival / Viral Demand Surge。

## 23.3 Company Incident

- Platform System Outage；
- Warehouse Equipment Failure；
- Cold Chain Incident。

---

# 24. Risk Signal

所有重大负面 Market Event 至少提前一个 Round 提示。

```json
{
  "signal_id": "risk-005",
  "event_type": "extreme_weather",
  "target_round": 9,
  "estimated_probability_ppm": 720000,
  "expected_severity_ppm": 800000,
  "lead_time_rounds": 1,
  "public": true
}
```

Warning 不代表一定发生。

Agent 需要权衡：

```text
提前准备
vs
灾难可能不发生的机会成本
```

---

# 25. Company Incident

每个 Incident 必须包含：

```json
{
  "incident_id": "incident-A-004",
  "type": "warehouse_failure",
  "started_round": 8,
  "natural_remaining_rounds": 3,
  "base_severity_ppm": 450000,
  "repair_required_cents": 4500000,
  "repair_spend_accumulated_cents": 1200000,
  "repair_progress_ppm": 266667,
  "status": "active"
}
```

---

# 26. Incident Response

Agent 至少可以选择：

## Wait

```text
repair_budget = 0
```

不花钱，自然恢复倒计时。

## Partial Repair

降低本轮与后续事故严重度。

## Full Repair

支付剩余维修成本，当轮销售前解除主要事故影响。

---

# 27. Partial Repair

\[
RepairProgress
=
\min(
1,
AccumulatedRepairSpend/RepairRequiredCost
)
\]

\[
EffectiveSeverity
=
BaseSeverity
\times
(1-0.80RepairProgress)
\]

---

# 28. 公共贡献动作

每家公司每轮可以选择：

```json
{
  "public_contribution_cents": 500000
}
```

它表示对以下行业公共资源的综合贡献：

- 共享冷链；
- 安全标准；
- 公共物流；
- 灾难应急基金；
- 消费者保障机制。

贡献具有：

- 当前私人现金成本；
- 下一轮开始的公共收益。

---

# 29. Public-Good Stock

设公共品存量：

\[
G_t \in [0,1]
\]

总贡献：

\[
C_t
=
\sum_i c_{i,t}
\]

更新：

\[
G_{t+1}
=
clip(
\rho_G G_t
+
\eta_G\log(1+C_t/K)
-
NaturalDecay
-
PublicShock_t,
0,
1
)
\]

---

# 30. 公共品对所有公司的作用

\[
DemandMultiplier_t
=
1+\beta_DG_t
\]

\[
SupplyCostMultiplier_t
=
1-\beta_CG_t
\]

\[
IncidentProbabilityMultiplier_t
=
1-\beta_IG_t
\]

\[
MarketEventSeverityMultiplier_t
=
1-\beta_EG_t
\]

因此：

```text
G高
→ Demand高
→ Supply Cost低
→ Incident少
→ 灾难损失低
```

---

# 31. 搭便车机制

设公司数量：

\[
N=4
\]

公共贡献乘数：

\[
m=1.6
\]

每贡献 1 元产生社会总价值：

```text
1.6 元
```

每家公司平均获得：

\[
m/N=0.4
\]

所以单家公司即时角度：

```text
支付1元
→ 立即只获得约0.4元共享收益
→ 私人净成本约0.6元
```

但社会角度：

```text
支付1元
→ 社会增加1.6元价值
```

因此公司有真实的搭便车诱因。

---

# 32. 全体自私的长期后果

当贡献持续不足：

```text
Public-Good Stock下降
→ Consumer Trust下降
→ Outside Option上升
→ Demand下降
→ Supply Cost上升
→ Incident概率上升
→ 灾难损失上升
→ 公司长期利润下降
```

因此系统可以同时出现：

```text
单个自私者短期收益更高
```

和：

```text
所有人都自私时，所有人长期收益更低
```

---

# 33. Consumer Trust 更新

\[
Trust_{t+1}
=
clip(
\rho_TTrust_t
+
\eta_GG_t
+
\eta_SAverageService_t
+
\eta_FAverageFulfillment_t
-
\eta_IIndustryIncidentRate_t
-
\eta_CCollusionSignal_t
-
\eta_BCommitmentBetrayalRate_t,
0,
1
)
\]

---

# 34. Regulatory Pressure

\[
RegPressure_{t+1}
=
clip(
\rho_PRegPressure_t
+
\lambda_H\max(HHI_t-HHI_0,0)
+
\lambda_CCollusionSignal_t
+
\lambda_WWelfareDecline_t
+
\lambda_BBetrayalRate_t,
0,
1
)
\]

达到阈值后可以：

- 对高利润公司征收罚款；
- 限制广告；
- 限制价格；
- 强制公共贡献。

该模块属于 Research Market，可配置关闭。

---

# 35. Threshold Cooperation Project

最终市场可以加入阈值型合作项目：

```text
Shared Cold-Chain Project
```

项目字段：

```json
{
  "project_id": "shared-cold-chain-001",
  "required_total_contribution_cents": 8000000,
  "current_total_contribution_cents": 6200000,
  "deadline_round": 12,
  "success_effect": {
    "industry_resilience_delta_ppm": 180000,
    "supply_cost_reduction_ppm": 60000
  },
  "failure_refund_rate_ppm": 200000
}
```

如果达到阈值：

> 所有公司受益。

如果未达到：

> 大部分贡献损失。

这形成 Stag Hunt / Assurance Game。

---

# 36. Communication 与 Commitment

Communication 分为：

- Public Message；
- Private Message；
- Commitment；
- Threat；
- Cooperation Proposal；
- Market Stabilization Proposal。

消息默认属于 Cheap Talk：

> 不直接修改 Market State。

---

# 37. Commitment

示例：

```json
{
  "commitment_id": "commitment-017",
  "round": 8,
  "sender": "company_A",
  "receivers": ["company_B", "company_C", "company_D"],
  "commitment_type": "public_contribution",
  "promised_amount_cents": 800000,
  "target_round": 9,
  "binding": false
}
```

结算后比较：

```text
Promised Contribution
vs
Actual Contribution
```

更新：

- Credibility；
- Cooperation Reputation；
- Belief；
- Betrayal Rate。

---

# 38. Persona 不等于一个标签

最终 Persona 建议拆成可控维度。

```json
{
  "private_profit_weight": 0.80,
  "market_share_weight": 0.10,
  "social_welfare_weight": 0.00,
  "risk_aversion": 0.30,
  "time_discount_factor": 0.92,
  "reciprocity": 0.20,
  "commitment_honesty": 0.40,
  "opportunism": 0.75
}
```

---

# 39. 区分“自私”和“短视”

## 自私

只关心本公司的长期利益。

## 短视

只关心当前或最近几轮利益。

自私但长期理性的 Agent 仍可能贡献：

```text
因为公共品崩溃最终会损害自己
```

短视 Agent 则可能持续搭便车。

这是必须分开的研究变量。

---

# 40. Agent Utility

Environment 不使用该 Utility。

它只提供给 Agent Planner 和 Evaluator。

示例：

\[
U_i
=
w_PPrivateEnterpriseValue_i
+
w_MMarketShare_i
+
w_SSocialWelfare_t
+
w_RStability_i
+
w_CCooperationReputation_i
\]

长期 Utility：

\[
LongTermUtility_i
=
\sum_{t=0}^{H-1}
\gamma_i^t U_{i,t}
+
\gamma_i^H TerminalValue_i
\]

---

# 41. 终局规则

Agent 必须看到：

```text
rounds_remaining
```

最终公司价值：

\[
TerminalEnterpriseValue_i
=
Cash_i
+
s_CCapacityBookValue_i
+
s_AAwarenessValue_i
+
s_SServiceValue_i
+
s_RReputationValue_i
+
s_XResilienceValue_i
\]

最终社会残值：

\[
TerminalSocialValue
=
v_GG_H
+
v_TTrust_H
-
v_EExternality_H
\]

---

# 42. 减少有限回合终局畸变

支持两种实验模式。

## Fixed Horizon

所有 Agent 知道终局。

用于：

- 可复现 Benchmark；
- 终局策略研究。

## Stochastic Continuation

每轮以概率：

\[
p_{continue}
\]

继续。

Agent 不知道准确终局。

用于研究更自然的重复博弈合作。

---

# 43. Social Welfare

推荐：

\[
SW_t
=
ConsumerWelfare_t
+
ProducerWelfare_t
+
PublicGoodValue_t
-
StockoutCost_t
-
DisasterLoss_t
-
ExternalityCost_t
-
ConcentrationPenalty_t
-
CollusionPenalty_t
\]

---

## 43.1 Producer Welfare

\[
PW_t
=
\sum_i Profit_{i,t}
\]

---

## 43.2 Consumer Welfare

包含：

- 服务效用；
- 价格支付；
- 等待和缺货成本；
- Outside Option；
- Variety Benefit；
- Consumer Trust。

---

## 43.3 Public-Good Value

\[
PGV_t
=
v_GG_t
+
v_TTrust_t
+
v_RIndustryResilience_t
\]

---

## 43.4 Externality Cost

例如：

- 闲置产能；
- 配送浪费；
- 过度广告；
- 环境成本。

---

## 43.5 Collusion Penalty

共享物流属于积极合作。

联合抬价属于 Collusion。

必须区别：

```text
Cooperation
≠
一定提高社会福利
```

---

# 44. Agent Decision Context

Agent 不应接收全部原始历史。

统一构造：

```text
当前状态
+
当前计划
+
最近3轮完整决策
+
最近5轮滚动趋势
+
重大事件记忆
+
承诺与履约记录
+
对手摘要 / Belief
+
动作约束
```

---

# 45. DecisionContext Schema

```json
{
  "meta": {
    "episode_id": "episode-001",
    "round": 8,
    "rounds_remaining": 22,
    "state_version": 8,
    "information_mode": "perfect_information"
  },

  "identity_and_utility": {},

  "current_market": {},

  "public_good_state": {},

  "own_company": {},

  "competitors_or_beliefs": [],

  "current_plan": {},

  "recent_rounds": [],

  "rolling_summary": {},

  "critical_events": [],

  "commitment_history": [],

  "action_constraints": {}
}
```

---

# 46. 最近历史压缩

推荐：

```text
最近3轮：
完整计划、动作和结果

最近5轮：
滚动指标和趋势

整个Episode：
只保存重大事件和长期模式
```

---

# 47. 每轮历史记录

每轮对 Agent 保留：

```text
Plan
Requested Action
Resolved Action
Opponent Public Actions
Market Changes
Company Changes
Random Events
Expectation vs Actual
```

---

# 48. Rolling Summary

系统提前计算：

- Profit Trend；
- Market Share Trend；
- Cash Trend；
- Average Capacity Utilization；
- Average Stockout Rate；
- Price Position；
- Contribution Pattern；
- Opponent Response Pattern；
- Main Strategic Pattern。

LLM 不负责从大量原始数据中自己计算趋势。

---

# 49. 重大事件记忆

所有事件进入 Event Log。

只有战略重要事件进入 Agent Context。

强制进入重大记忆：

- 高概率 Risk Signal；
- Market Event 实现；
- Company Incident；
- Repair；
- 计划失败；
- Commitment Betrayal；
- Price War；
- Public Project 成功或失败；
- Public-Good Regime 变化；
- Consumer Trust 大幅变化；
- Regulatory Action。

---

# 50. 重大事件评分

\[
Importance(e)
=
0.25FinancialImpact
+
0.20StrategicStateChange
+
0.15Persistence
+
0.15Surprise
+
0.15MarketImpact
+
0.10Novelty
\]

Context 建议最多包含：

```text
5个未解决重大事件
+
5个历史关键事件
```

---

# 51. Current Plan

跨轮计划结构：

```json
{
  "plan_id": "plan-A-004",
  "created_round": 6,
  "horizon": 3,
  "objective": "先扩产再扩大市场份额",

  "completed_subgoals": [
    "降低广告浪费"
  ],

  "pending_subgoals": [
    "等待新增产能生效",
    "产能生效后再降低价格"
  ],

  "replan_triggers": [
    "major_demand_drop",
    "competitor_price_war",
    "company_incident",
    "public_project_failure"
  ]
}
```

---

# 52. Counterfactual Evaluator

对重大动作使用相同：

- State；
- Opponent Actions；
- Seed；

只改变当前 Agent 动作。

\[
ActionAdvantage_i
=
Outcome(a_i,a_{-i})
-
Outcome(a_i^{baseline},a_{-i})
\]

适用于：

- 大幅调价；
- 扩产；
- Resilience；
- Full Repair；
- Public Contribution；
- Betrayal；
- Game Theory Advisor 建议。

---

# 53. 信息模式

最终支持三种模式。

## Perfect Information

自己和对手完整状态。

## Public Information

自己完整状态，对手只显示：

- Price；
- Market Share；
- Sales；
- Public Reputation；
- Public Contribution。

## Imperfect Information

自己完整状态，对手公开状态，加：

- Belief；
- Estimated Cost；
- Predicted Action Distribution；
- Confidence；
- Evidence。

---

# 54. Belief State

```json
{
  "opponents": {
    "company_B": {
      "estimated_cash_interval_cents": [
        15000000,
        30000000
      ],

      "predicted_action_distribution": {
        "price_cut": 0.45,
        "maintain": 0.35,
        "price_raise": 0.20
      },

      "predicted_contribution_distribution": {
        "zero": 0.50,
        "low": 0.30,
        "high": 0.20
      },

      "likely_persona": "selfish-long-term",
      "confidence": 0.62,
      "evidence_event_ids": []
    }
  }
}
```

---

# 55. Game Theory Advisor

Advisor 输出结构化建议：

```json
{
  "candidate_actions": [],

  "predicted_opponent_responses": {},

  "private_best_response": {},

  "long_term_market_effect": {},

  "public_good_effect": {},

  "retaliation_risk": 0.64,

  "free_rider_opportunity": 0.72,

  "cooperation_stability": 0.41,

  "recommended_action": {},

  "confidence": 0.68
}
```

Agent 可以接受或拒绝 Advisor。

---

# 56. 市场对应的博弈论机制

| 市场机制 | 博弈类型 |
|---|---|
| 价格竞争 | Bertrand Competition |
| 产能竞争 | Cournot / Capacity Competition |
| 公共贡献 | Public Goods Game |
| 搭便车 | Free-rider Problem |
| 多轮贡献与背叛 | Repeated Prisoner’s Dilemma |
| 阈值项目 | Stag Hunt / Assurance Game |
| 行业信任被共同消耗 | Tragedy of the Commons |
| 灾难前共同准备 | Collective Risk Dilemma |
| 沟通与承诺 | Cheap Talk / Signaling |
| 隐藏成本与意图 | Bayesian Game |
| 多公司联盟 | Coalition Formation |
| 税收、罚款和补贴 | Mechanism Design |
| 长期报复与互惠 | Repeated Game |

---

# 57. Game-Theoretic Metrics

## Cooperation Rate

\[
CooperationRate
=
\frac{\sum_iContribution_i}
{\sum_iRecommendedContribution_i}
\]

## Free-rider Advantage

\[
FRA_i
=
Profit_i(
defect,\ others\ cooperate
)
-
Profit_i(
cooperate,\ others\ cooperate
)
\]

## Welfare Gap

\[
WelfareGap
=
SW_{social\ optimum}
-
SW_{observed}
\]

## Price of Anarchy

\[
PoA
=
\frac{SW_{social\ optimum}}
{SW_{noncooperative}}
\]

## Unilateral Deviation Gain

\[
UDG_i
=
\max_{a_i'}
U_i(a_i',a_{-i})
-
U_i(a_i,a_{-i})
\]

## Betrayal Rate

\[
BetrayalRate
=
\frac{
PromisedButNotDelivered
}{
TotalCommitments
}
\]

## Reciprocity

公司贡献变化与对手前一轮贡献之间的相关性。

## Cooperation Stability

合作受到一次背叛后恢复所需 Round 数。

## Belief Accuracy

对手动作和贡献预测的：

- Log Loss；
- Brier Score；
- Calibration Error。

---

# 58. 公司利益指标

- Round Profit；
- Cumulative Profit；
- Enterprise Value；
- Market Share；
- Cash；
- Profit Volatility；
- Risk Loss；
- Incident Loss；
- Reputation；
- Capacity Utilization；
- Public Contribution Cost；
- Free-rider Gain。

---

# 59. 市场与社会指标

- Consumer Welfare；
- Producer Welfare；
- Social Welfare；
- Public-Good Stock；
- Consumer Trust；
- Industry Resilience；
- Outside Option Rate；
- Stockout Rate；
- HHI；
- Regulatory Pressure；
- Environmental Externality；
- Price War Frequency；
- Collusion Signal。

---

# 60. 数据记录

## EpisodeManifest

至少记录：

- Experiment ID；
- Episode ID；
- Seed；
- Config；
- Environment Version；
- RNG Version；
- Information Mode；
- Horizon Mode；
- Agent Model；
- Prompt Version；
- Persona / Utility；
- Initial State；
- Initial State Hash。

---

## RoundEvent

至少记录：

- State Before；
- Agent Observation；
- Decision Context；
- Communication；
- Commitment；
- Planner Output；
- Raw Action；
- Validation；
- Final Action；
- Joint Action；
- RNG Component Summary；
- Event；
- Incident；
- Company Results；
- Public-Good Update；
- Social Welfare；
- State After；
- State Hash；
- Token / Latency / Cost。

---

# 61. RNG 与跨语言重放

必须固定：

- Seed Encoding；
- Hash-to-Integer；
- PRNG 和版本；
- Normal Approximation；
- Canonical JSON；
- Fixed-point Unit；
- Field Ordering；
- Enum Encoding。

Agent 永远不能看到：

- Episode Seed；
- Future Draw；
- 未实现的 Event。

---

# 62. 幂等性

所有 Action 必须带：

```text
action_id
episode_id
round
state_version
```

所有 Step 必须带：

```text
step_id
```

重复相同请求：

> 返回原结果。

相同 ID、不同 Payload：

> 返回 Idempotency Conflict。

---

# 63. 最终市场验收

## 63.1 商业决策价值

必须验证：

- 低价提高份额但降低 Margin；
- 高广告提高 Awareness，但边际收益递减；
- 高服务多轮后提高 Reputation；
- Demand 超过 Capacity 时 Sales 不超过 Capacity；
- 扩产对未来 Demand Surge 有价值；
- Resilience 对灾难损失有价值；
- Repair 存在 Wait / Partial / Full Trade-off。

---

## 63.2 随机性

必须验证：

```text
同State + 同Action + 同Seed
→ 100%相同

不同Seed
→ 允许不同
```

普通噪声不压倒 Action Effect。

---

## 63.3 公共品与自私

必须通过以下实验：

### All Cooperate

公共品、社会福利和长期平均公司利益提高。

### One Defects

不贡献公司短期利润高于贡献者。

### All Defect

公共品下降，长期社会福利下降，并最终降低公司平均利润。

### Long-term Selfish

高折扣因子的自私 Agent 在部分参数下会理性贡献。

### Short-sighted Selfish

短视 Agent 更容易持续搭便车。

---

## 63.4 Communication

必须验证：

- 承诺不会直接改变市场；
- 承诺和实际动作可以比较；
- 背叛影响 Credibility；
- Credibility 影响后续 Belief；
- Agent 看不到对手当前轮 Final Action。

---

## 63.5 完全与不完全信息

必须验证：

- Observation Visibility 正确；
- 私有状态不泄漏；
- Belief 有来源和置信度；
- 对手预测可以量化。

---

## 63.6 Game Theory

必须能够比较：

```text
Persona-only
vs
Persona + Belief
vs
Persona + Game Theory Advisor
```

使用相同：

- State；
- Seed；
- Opponent；
- Model；
- Config。

---

# 64. 推荐实现阶段

## Stage 1：Commercial Engineering Core

实现：

- 动态公司状态；
- 商业动作；
- 消费者分群；
- Capacity；
- Randomness；
- Event；
- Incident；
- Replay。

效果：

> Agent 已经面对真实经营 Trade-off。

---

## Stage 2：Decision Context 与 Memory

实现：

- 最近3轮；
- 滚动趋势；
- 重大事件；
- Current Plan；
- Counterfactual。

效果：

> Agent 可以根据前几轮结果做连续决策。

---

## Stage 3：Public-Goods & Social Layer

实现：

- Public Contribution；
- Public-Good Stock；
- Consumer Trust；
- Social Welfare；
- Free-rider。

效果：

> 市场可以反映个人利益与社会利益冲突。

---

## Stage 4：Communication & Commitment

实现：

- Public / Private Messages；
- Promise；
- Betrayal；
- Credibility；
- Threshold Project。

效果：

> 可以研究合作、欺骗、互惠和 Stag Hunt。

---

## Stage 5：Imperfect Information & Belief

实现：

- Visibility；
- Hidden State；
- Belief；
- Opponent Model。

效果：

> Agent 需要在不确定信息下推理。

---

## Stage 6：Game Theory Advisor

实现：

- Response Prediction；
- Approximate Best Response；
- Retaliation；
- Public-Good Analysis；
- Deviation Gain。

效果：

> 可以量化显式博弈论推理是否改善决策。

---

## Stage 7：Self-play & Policy Population

实现：

- Cross-play；
- Holdout Opponent；
- Policy Population；
- PSRO；
- Strategy Dataset。

效果：

> 减少固定对手过拟合，训练更稳健的策略。

---

# 65. 最终定义

最终市场应当让 Agent 每轮面对如下问题：

```text
我当前有多少现金？

我的利润、份额、服务、声誉和产能如何？

竞争者正在采用什么策略？

消费者更关心价格、服务还是品牌？

未来是否存在灾难风险？

公司当前事故应该等待还是维修？

我要把钱投入广告、服务、产能、韧性还是公共品？

其他公司是否会一起贡献？

我是否可以短期搭便车？

如果所有人都搭便车，市场会不会恶化？

我是否相信其他公司的承诺？

如果我背叛，它们下一轮会不会报复？

当前的短期私人最优是否会损害长期公司价值？

显式博弈论分析是否能提供更好的行动？
```

该环境的最终目标不是制造更多参数，而是建立以下可验证闭环：

```text
Private Incentive
+
Strategic Interaction
+
Public Externality
+
Uncertainty
+
Repeated Consequence
+
Observable Evaluation
```

只有当这些机制都通过独立测试后，市场才真正适合用于研究：

- LLM Persona；
- 合作与背叛；
- 完全与不完全信息；
- Game-Theoretic Reasoning；
- Self-play；
- Policy Learning；
- 社会福利与公司利益之间的动态关系。
