# LLM 多智能体生鲜配送市场博弈系统
## Engineering MVP 技术规格 v4.0

**配套权威配置：** `market_v4.yaml`  
**状态：** 可作为后端代码骨架、测试和前端接口的统一实现依据。

---

# 0. 本次修订解决什么问题

v4.0 在动态市场 v3 的基础上补齐了开始完整实现前必须明确的工程契约：

1. 唯一、可加载的默认市场配置；
2. 无歧义的回合时序；
3. 不会出现逻辑空洞的现金约束；
4. 可跨实现复核的 RNG 与 State Hash 协议；
5. 消费者“不购买”外部选项；
6. 缺货后的确定性一次转售规则；
7. 终局、残值、折旧和最后一轮投资规则；
8. `action_id`、`step_id` 和幂等执行；
9. `EpisodeManifest` 与完整重放信息；
10. Persona Utility 的归一化方法；
11. 连续动作的动态约束接口；
12. 后端是唯一市场计算源，前端不再复制公式。

---

# 1. 规范来源与版本优先级

项目必须只保留一个市场规则来源。

优先级如下：

```text
1. market_v4.yaml
   └── 所有默认参数、边界和事件分布的唯一权威来源

2. 本文档 Engineering MVP v4.0
   └── 算法顺序、数据结构、接口、RNG、哈希、幂等和验收契约

3. 完整项目报告 v3.0
   └── 研究问题、Research MVP、博弈论、自博弈和数据训练设计

4. 旧版 v2 / v3 / v2.1 文档
   └── 仅作历史记录，不再作为实现规范
```

任何代码常量不得重新抄写 `market_v4.yaml` 中的参数。

---

# 2. Engineering MVP 的目标

Engineering MVP 只回答：

> 系统是否已经具备一个正确、可复现、可扩展、具有真实决策 Trade-off 的多 Agent 市场环境？

它不要求证明：

- LLM 比 Rule Agent 更强；
- Persona 一定导致显著差异；
- Game Theory Advisor 一定提高收益；
- 不完全信息一定更困难。

它必须证明：

```text
Company State
+ Market State
+ Risk Signal / Incident
        ↓
Agent Numeric Action
        ↓
Joint Action + Action Lock
        ↓
MarketEnv.step()
        ↓
Result + Next State
        ↓
Event Log + Replay + Evaluation
```

---

# 3. MVP 实现边界

## 3.1 必须实现

- 2～4 家公司，接口可扩展到 8 家；
- 完全信息；
- 公司独立财务、经营、品牌和风险状态；
- 连续数值动作；
- Preset 到 Numeric Action 的解析；
- 三类消费者与外部选项；
- 产能约束；
- 一次缺货转售；
- 现金不为负的流动性规则；
- 广告、服务、声誉、产能和韧性的跨轮状态；
- 市场级风险信号与重大事件；
- 公司级事故与主动维修；
- 稳定 Seed 派生 RNG；
- 幂等 Action / Step；
- Event Log、EpisodeManifest、Replay；
- Rule、Mock、LLM Agent；
- Evaluator 与 View 分离。

## 3.2 暂不实现

- 不完全信息；
- Belief State；
- Agent 间真实谈判；
- Cooperative Persona；
- 社会福利进入 Agent 目标；
- Nash / Best Response / PSRO；
- 贷款、融资和公司退出；
- 前端独立市场计算。

---

# 4. 单位、精度与 Canonical State

为避免 Python、TypeScript、数据库和日志之间出现精度漂移，持久化状态不保存任意浮点数。

| 类型 | 存储单位 | 示例 |
|---|---|---|
| 金额 | 整数分 `cent` | `10000 = 100.00` |
| 比例 | 整数 `ppm` | `250000 = 25%` |
| 分数 | 整数 `ppm` | `700000 = 0.7` |
| 销量 | 整数订单 | `2451` |
| Round | 整数 | `1` |
| Duration | 整数回合 | `2` |

计算过程中允许后端使用高精度 Decimal 或 float64，但写入 Canonical State 前必须：

1. 按 `market_v4.yaml` 指定精度量化；
2. 使用 half-even；
3. 转换为整数固定点；
4. 再生成 State Hash。

前端只把固定点值转换为显示值，不参与状态计算。

---

# 5. 权威默认配置

所有参数来自：

```text
market_v4.yaml
```

配置必须在 Episode 创建时复制一份快照或记录：

```text
config_id
config_version
config_sha256
```

运行中的 Episode 不允许热修改配置。

核心默认值包括：

| 参数 | 默认值 |
|---|---:|
| 公司数 | 4 |
| 回合数 | 10 |
| Base Demand | 12000 orders |
| 初始 Cash | 300000.00 |
| 初始 Capacity | 3500 orders |
| 初始 Unit Cost | 60.00 |
| 初始 Awareness | 0.35 |
| 初始 Service | 0.55 |
| 初始 Reputation | 0.50 |
| 初始 Resilience | 0.15 |
| Price Range | 75.00～130.00 |

完整参数不在本文重复维护，避免双源漂移。

---

# 6. State 模型

## 6.1 MarketState

```json
{
  "episode_id": "episode-0001",
  "round": 4,
  "rounds_remaining": 7,
  "state_version": 3,
  "terminal": false,

  "market": {
    "base_demand_orders": 12000,
    "realized_demand_orders": 12640,
    "no_purchase_orders": 830,
    "lost_after_stockout_orders": 190,
    "market_sentiment_ppm": 1020000,
    "base_supply_cost_index_ppm": 1015000,
    "actual_supply_cost_index_ppm": 1319500,
    "average_paid_price_cents": 9780
  },

  "consumer_segments": {},
  "risk_signals": [],
  "active_market_events": [],
  "companies": {},
  "last_joint_action": {},
  "state_hash": "sha256:..."
}
```

### Round 语义

`state.round` 表示：

> 当前等待 Agent 决策并将被结算的回合。

初始状态：

```text
state.round = 1
state_version = 0
```

结算 Round 1 后：

```text
StepResult.settled_round = 1
next_state.round = 2
next_state.state_version = 1
```

结算最后一轮后：

```text
next_state.round = max_rounds + 1
next_state.terminal = true
```

---

## 6.2 CompanyState

```json
{
  "company_id": "company_A",
  "persona": "aggressive",

  "financial": {
    "cash_balance_cents": 25000000,
    "round_revenue_cents": 21000000,
    "round_variable_cost_cents": 13500000,
    "round_fixed_spend_cents": 3200000,
    "round_incident_cost_cents": 0,
    "round_profit_cents": 4300000,
    "cumulative_profit_cents": 11800000,
    "capacity_book_value_cents": 16200000
  },

  "commercial": {
    "price_cents": 9600,
    "market_share_ppm": 270000,
    "potential_demand_orders": 3420,
    "sales_orders": 2800,
    "attempted_unfulfilled_orders": 620,
    "orders_received_from_redistribution": 90,
    "orders_lost_after_redistribution": 530
  },

  "operations": {
    "base_capacity_orders": 3000,
    "capacity_pipeline": [],
    "effective_capacity_orders": 2800,
    "financial_capacity_orders": 999999999,
    "capacity_utilization_ppm": 1000000,
    "base_unit_cost_cents": 6000,
    "actual_unit_cost_cents": 7917
  },

  "brand": {
    "brand_awareness_ppm": 540000,
    "service_quality_ppm": 720000,
    "reputation_ppm": 680000,
    "last_attempted_unfulfilled_rate_ppm": 181287
  },

  "risk": {
    "resilience_ppm": 350000,
    "active_incident": null
  },

  "history": {
    "last_action_id": "...",
    "last_action": {},
    "recent_profit_cents": [],
    "recent_market_share_ppm": []
  }
}
```

---

# 7. Action 模型

## 7.1 Canonical Numeric Action

```json
{
  "action_id": "0195...",
  "episode_id": "episode-0001",
  "agent_id": "company_A",
  "round": 4,
  "state_version": 3,

  "price_cents": 9450,
  "advertising_budget_cents": 1800000,
  "service_budget_cents": 1200000,
  "capacity_investment_cents": 1500000,
  "resilience_budget_cents": 600000,

  "incident_response": {
    "mode": "partial_repair",
    "repair_budget_cents": 1200000
  },

  "strategy_summary": "..."
}
```

`strategy_summary` 不进入 Environment 计算。

## 7.2 Preset

Preset 只用于输入便利：

```json
{
  "price": "low",
  "advertising": "high",
  "service": "medium",
  "capacity": "medium",
  "resilience": "low"
}
```

必须由 `PresetResolver` 使用 `market_v4.yaml` 转换为 Canonical Numeric Action。

## 7.3 动态约束接口

连续动作空间不再提供 `legal_actions()`。

使用：

```python
constraints = env.get_action_constraints(
    agent_id=agent_id,
    state_version=state.state_version
)
```

返回：

```json
{
  "schema_version": "company-action-v4.0.0",
  "cash_available_cents": 25000000,
  "bounds": {},
  "capacity_investment_enabled": true,
  "resilience_investment_enabled": true,
  "active_incident": {},
  "max_useful_repair_budget_cents": 3300000,
  "constraints": [
    "total_fixed_spend <= cash_at_round_start"
  ]
}
```

最后一轮：

```text
capacity_investment_enabled = false
resilience_investment_enabled = false
```

因为二者默认下一轮生效。

---

# 8. 幂等协议

## 8.1 Action 幂等

每次提交必须有 `action_id`。

规则：

```text
相同 action_id + 相同 payload
→ 返回第一次校验/提交结果

相同 action_id + 不同 payload
→ IDEMPOTENCY_CONFLICT
```

## 8.2 Step 幂等

```text
step_id = episode_id:round:state_version
```

规则：

```text
相同 step_id + 相同 joint_action_hash
→ 返回缓存 StepResult

相同 step_id + 不同 joint_action_hash
→ IDEMPOTENCY_CONFLICT
```

## 8.3 State Version

Action 引用的版本与当前状态不一致：

```text
STATE_VERSION_CONFLICT
```

前端或 Controller 必须重新获取状态，不能静默覆盖。

---

# 9. 精确回合算法

本节是唯一回合时序。

定义：

```text
state_t + joint_action_t + deterministic_random_t
→ result_t + state_t+1
```

## 9.1 state_t 中已经确定的内容

Agent 决策前已经知道：

- 当前 Active Market Events；
- 当前公司 Incident；
- 针对未来回合的 Risk Signals；
- 当前 Company / Competitor State；
- `rounds_remaining`。

重大事件不会在 Agent 提交动作后突然插入本轮。

## 9.2 当前动作的生效时点

| 动作 | 本轮生效 | 下一轮生效 |
|---|---:|---:|
| Price | 是 | 新动作覆盖 |
| Advertising | 是 | Awareness 保留部分影响 |
| Service | 是 | Service / Reputation 保留影响 |
| Incident Repair | **销售前立即生效** | 若未完全修复则保留进度 |
| Capacity Investment | 否 | 是 |
| Resilience Investment | 否 | 是 |

因此：

- 当前轮购买的 Resilience **不能**抵挡已经 Active 的本轮灾难；
- 它可以抵挡下一轮及后续事件；
- 多轮灾难中，本轮投资可从下一轮开始降低剩余冲击；
- 当前事故维修先于本轮销售，所以 Agent 可以主动缩短损失。

## 9.3 算法步骤

```text
A. 验证 episode / round / state_version

B. 收集、校验并 Lock 所有 Final Actions

C. 扣除当前固定决策支出：
   Advertising + Service + Capacity Investment
   + Resilience Investment + Repair

D. 在销售前应用 Incident Repair：
   Wait / Partial / Full

E. 读取 state_t 中已经生效的 Base Capacity

F. 计算本轮 Active Event Effects：
   使用 state_t 开始时的 Resilience

G. 生成本轮确定性随机组件：
   Demand Noise / Consumer Noise / Operational Noise

H. 计算各 Segment 初始 Utility：
   所有公司 + Outside Option

I. 第一遍整数订单分配

J. 对每家公司应用：
   Operational Capacity + Financial Capacity

K. 按 Segment 比例履约，计算 Attempted Unfulfilled

L. 对缺货订单进行一次重新分配：
   有剩余 Capacity 的公司 + Outside Option

M. 剩余未满足订单变为 Lost Demand

N. 计算 Revenue / Variable Cost / Refund / Profit / Cash

O. 更新 Awareness / Service / Reputation

P. 更新下一轮 Capacity / Resilience / Book Value

Q. 推进 Active Event 和 Incident 自然倒计时

R. 根据当前 Warning 判断下一轮 Major Event 是否 Realize

S. 为后续回合生成新的 Risk Signals

T. 为下一轮采样 Company Incident

U. 生成 next_state、State Hash 和 StepResult
```

---

# 10. 权威市场计算公式

本节定义 `MarketEnv.step()` 使用的公式。参数值全部从 `market_v4.yaml` 读取。

## 10.1 固定点辅助函数

所有 ratio / score 最终保存为 ppm。

定义：

\[
clip(x,l,h)=\min(\max(x,l),h)
\]

预算饱和函数：

\[
Sat(b,s)=\frac{b}{b+s}
\]

其中：

- \(b\)：预算；
- \(s\)：配置中的 Scale。

`Sat` 范围 `[0,1)`，避免广告、服务和韧性投入线性无限增长。

## 10.2 本轮广告、服务和韧性输入

\[
AdInput_i=Sat(AdBudget_i,AdScale)
\]

\[
ServiceInput_i=Sat(ServiceBudget_i,ServiceScale)
\]

\[
ResilienceInput_i=Sat(ResilienceBudget_i,ResilienceScale)
\]

所有结果量化为 ppm。

## 10.3 当前报价均价与相对价格信号

消费者选择前，使用所有公司当前 Action 中的报价计算算术平均：

\[
OfferedAveragePrice_t=\frac{1}{N}\sum_i Price_i
\]

公司 \(i\) 的相对价格信号：

\[
RelativePriceSignal_i
=
\frac{OfferedAveragePrice_t-Price_i}{PriceScale}
\]

保存前 Clamp 到 `[-2,2]`，再转为 score ppm。

价格低于竞争者时为正，高于竞争者时为负。

## 10.4 当前 Market Event 的实际影响

对于一个不利乘数 \(m<1\)，原始损失：

\[
Loss=1-m
\]

若该 Event 允许 Resilience：

\[
ProtectedLoss_i
=
Loss\times(1-MaxReduction\times Resilience_{i,t})
\]

\[
ProtectedMultiplier_i=1-ProtectedLoss_i
\]

对于成本上涨乘数 \(m>1\)：

\[
Increase=m-1
\]

\[
ProtectedCostMultiplier_i
=
1+Increase\times(1-MaxReduction\times Resilience_{i,t})
\]

对于 Service Penalty：

\[
ProtectedPenalty_i
=
RawPenalty\times(1-MaxReduction\times Resilience_{i,t})
\]

Resilience 不削弱正向 Demand Surge。

## 10.5 公司事故 Repair 后的影响

本轮 Repair 先累计：

\[
RepairProgress_i
=
\min\left(1,
\frac{AccumulatedRepair_i+RepairBudget_i}
{RepairRequiredCost_i}
\right)
\]

若 `RepairProgress = 1`：

```text
Incident 在本轮 Demand / Sales 前标记为 resolved，所有事故影响归零。
```

否则定义：

\[
IncidentImpactFactor_i
=
1-MaxRepairMitigation\times RepairProgress_i
\]

对于事故损失乘数 \(m<1\)：

\[
EffectiveIncidentMultiplier_i
=
1-(1-m)\times IncidentImpactFactor_i
\]

对于事故 Penalty / Refund Rate：

\[
EffectivePenalty_i
=
RawPenalty\times IncidentImpactFactor_i
\]

## 10.6 有效广告与服务信号

Market Event 和 Company Incident 可能影响广告转化：

\[
EffectiveAdInput_i
=
AdInput_i
\times
MarketAdMultiplier_i
\times
IncidentAdMultiplier_i
\]

消费者当轮看到的 Awareness：

\[
ChoiceAwareness_i
=
clip(
PriorAwarenessWeight\times Awareness_{i,t}
+
CurrentAwarenessWeight\times EffectiveAdInput_i,
0,1)
\]

消费者当轮感受到的 Service：

\[
ChoiceService_i
=
clip(
PriorServiceWeight\times ServiceQuality_{i,t}
+
CurrentServiceWeight\times ServiceInput_i
-
MarketServicePenalty_i
-
IncidentServicePenalty_i,
0,1)
\]

## 10.7 Segment Utility

对于 Segment \(s\)：

\[
V_{i,s}
=
\beta^p_s PriceSignal_i
+
\beta^a_s ChoiceAwareness_i
+
\beta^q_s ChoiceService_i
+
\beta^r_s Reputation_{i,t}
+
\beta^o_s LastUnfulfilledRate_{i,t}
+
\epsilon_{i,s,t}
\]

所有 \(\beta\) 来自 `market_v4.yaml`；`prior_stockout` 权重为负。

Outside Option 使用配置中的固定：

\[
V_{out,s}=OutsideUtility_s
\]

## 10.8 Segment Choice

\[
P(i|s)
=
\frac{\exp(V_{i,s}/\tau)}
{\exp(V_{out,s}/\tau)+\sum_j\exp(V_{j,s}/\tau)}
\]

\[
P(out|s)
=
\frac{\exp(V_{out,s}/\tau)}
{\exp(V_{out,s}/\tau)+\sum_j\exp(V_{j,s}/\tau)}
\]

理论订单通过 Largest Remainder 转为整数。

## 10.9 本轮需求

当前状态包含 `market_sentiment_t` 和 `base_supply_cost_index_t`。

本轮实际需求：

\[
RealizedDemand_t
=
round\left(
BaseDemand
\times MarketSentiment_t
\times EventDemandMultiplier_t
\times(1+DemandNoise_t)
\right)
\]

所有比例在公式中先从 ppm 转为实数。

每个 Segment：

\[
Demand_{s,t}
=
LargestRemainder(
RealizedDemand_t\times SegmentWeight_s)
\]

## 10.10 当前供应成本

公司 \(i\) 的实际单位成本：

\[
ActualUnitCost_{i,t}
=
round\left(
BaseUnitCost_i
\times BaseSupplyCostIndex_t
\times ProtectedEventSupplyCostMultiplier_{i,t}
\right)
\]

## 10.11 Effective Capacity

\[
EffectiveCapacity_{i,t}
=
\left\lfloor
BaseCapacity_{i,t}
\times EventCapacityMultiplier_{i,t}
\times IncidentCapacityMultiplier_{i,t}
\times OperationalNoise_{i,t}
\right\rfloor
\]

最小为 0。

`BaseCapacity_{i,t}` 在 state_t 中已经是本轮生效产能；本轮新投资只进入 `BaseCapacity_{i,t+1}`。

## 10.12 Financial Capacity

当前固定支出：

\[
FixedSpend_i
=
Ad+Service+CapacityInvestment+ResilienceInvestment+Repair
\]

\[
CashAfterFixed_i=Cash_{i,t}-FixedSpend_i
\]

每单预计退款：

\[
ExpectedRefundPerOrder_i
=
Price_i\times EffectiveRefundRate_i
\]

每单贡献：

\[
Contribution_i
=
Price_i-ActualUnitCost_i-ExpectedRefundPerOrder_i
\]

如果 \(Contribution_i\ge0\)：

```text
FinancialCapacity = +∞
```

否则：

\[
FinancialCapacity_i
=
\left\lfloor
\frac{CashAfterFixed_i}{-Contribution_i}
\right\rfloor
\]

## 10.13 第一遍履约

公司总可履约上限：

\[
FulfillmentCap_i
=
\min(EffectiveCapacity_i,FinancialCapacity_i)
\]

若 Segment 初始分配总量超过该上限，则按各 Segment 初始订单比例分配公司可履约订单，并使用 Largest Remainder 整数闭合。

## 10.14 一次转售

按本文第 14 节规则，对 Attempted Unfulfilled 做一次重新选择和整数分配。

最终：

\[
Sales_i
=InitialFulfilled_i+RedistributedReceived_i
\]

## 10.15 收入、成本与现金

\[
Revenue_i=Price_i\times Sales_i
\]

\[
VariableCost_i=ActualUnitCost_i\times Sales_i
\]

\[
IncidentRefund_i
=Revenue_i\times EffectiveRefundRate_i
\]

\[
Profit_i
=Revenue_i-VariableCost_i-FixedSpend_i-IncidentRefund_i
\]

\[
Cash_{i,t+1}=Cash_{i,t}+Profit_i
\]

Financial Capacity 保证：

\[
Cash_{i,t+1}\ge0
\]

## 10.16 Awareness 更新

\[
Awareness_{i,t+1}
=
clip(
AwarenessRetention\times Awareness_{i,t}
+
AwarenessInputWeight\times EffectiveAdInput_i,
0,1)
\]

## 10.17 Service Quality 更新

\[
ServiceQuality_{i,t+1}
=
clip(
ServiceRetention\times ServiceQuality_{i,t}
+
ServiceInputWeight\times ServiceInput_i
-
MarketServicePenalty_i
-
IncidentServicePenalty_i,
0,1)
\]

## 10.18 Reputation 更新

公司原始尝试缺货率：

\[
AttemptedUnfulfilledRate_i
=
\frac{AttemptedUnfulfilled_i}{InitialAssignedDemand_i}
\]

若 Initial Assigned Demand 为 0，则为 0。

\[
FulfillmentRate_i=1-AttemptedUnfulfilledRate_i
\]

\[
Reputation_{i,t+1}
=
clip(
ReputationRetention\times Reputation_{i,t}
+
ReputationServiceWeight\times ChoiceService_i
+
ReputationFulfillmentWeight\times FulfillmentRate_i
-
MarketReputationPenalty_i
-
IncidentReputationPenalty_i,
0,1)
\]

即使消费者在第二遍转售成功，原公司仍承担首次履约失败的声誉影响。

## 10.19 Capacity 与 Book Value 更新

\[
CapacityAddition_i
=
\left\lfloor
CapacityInvestment_i/CapacityUnitCost
\right\rfloor
\]

\[
BaseCapacity_{i,t+1}
=
\left\lfloor
BaseCapacity_{i,t}\times(1-CapacityDepreciation)
\right\rfloor
+
CapacityAddition_i
\]

\[
BookValue_{i,t+1}
=
round(
BookValue_{i,t}\times(1-BookDepreciation)
)
+
CapacityInvestment_i
\]

## 10.20 Resilience 更新

本轮 Active Event 使用旧 Resilience。

下一轮：

\[
Resilience_{i,t+1}
=
clip(
ResilienceRetention\times Resilience_{i,t}
+
ResilienceInputWeight\times ResilienceInput_i,
0,1)
\]

## 10.21 Market Sentiment 更新

\[
Sentiment_{t+1}
=
clip(
(1-\alpha_s)Sentiment_t
+
\alpha_s SentimentMean
+
SentimentNoise_t,
SentimentMin,SentimentMax)
\]

## 10.22 Base Supply Cost Index 更新

\[
SupplyIndex_{t+1}
=
clip(
(1-\alpha_c)SupplyIndex_t
+
\alpha_c SupplyMean
+
SupplyNoise_t,
SupplyMin,SupplyMax)
\]

Active Supply Shock 只作用于本轮 `ActualSupplyCostIndex`，不永久写入 Base Index。

## 10.23 Event Signal 与 Realization

在完成 Round \(t\) 结算后：

1. 对 `target_round = t+1` 的既有 Signal 使用其 `estimated_probability` 采样 Realization；
2. Realized Event 进入 `state_{t+1}.active_market_events`；
3. 未 Realize 的 Signal 关闭并记录；
4. 对每个无 Active / Pending 同类事件的 Event Type，按配置采样是否生成新 Signal；
5. 新 Signal 的 Target：

\[
TargetRound=t+1+LeadTime
\]

6. 超过 Episode Horizon 的 Signal 不生成。

## 10.24 Company Incident 生成

仅当公司没有 Active Incident 时：

\[
P(Incident_i)
=
BaseIncidentP
\times
(1-IncidentProbabilityReduction\times Resilience_{i,t+1})
\]

事故类型和 Severity 按整数 ppm 权重采样。

Severity 影响再按：

\[
AdjustedImpact
=
RawImpact
\times
(1-IncidentSeverityReduction\times Resilience_{i,t+1})
\]

生成的 Incident 写入 `state_{t+1}`，Agent 下一轮可以在销售前主动维修。

---

# 11. 多事件组合规则

多个事件同时发生时，不按覆盖优先级处理，而按类型组合：

| 效果 | 组合方式 |
|---|---|
| Demand Multiplier | 相乘 |
| Supply Cost Multiplier | 相乘 |
| Capacity Multiplier | 每个事件先应用 Resilience，再相乘 |
| Advertising Conversion | 相乘 |
| Service Penalty | 相加后 clamp |
| Reputation Penalty | 相加后 clamp |

公司 Incident 与市场事件：

- Capacity Multiplier 相乘；
- Service / Reputation Penalty 相加；
- Refund Rate 使用公司事故值；
- 所有最终比例 Clamp 到配置边界。

MVP 限制：

- 同一类型 Market Event 不并发；
- 同一公司最多一个 Active Incident；
- 同时 Active Market Events 最多两个。

---

# 12. 消费者外部选项

旧 Softmax 会把全部需求强行分配给企业，这是不合理的。

v4 对每个 Segment 同时计算：

```text
Company A
Company B
...
Outside Option（不购买 / 延迟购买 / 转向线下）
```

对于 Segment `s`：

\[
P(i|s)=
\frac{\exp(V_{i,s}/\tau)}
{\exp(V_{out,s}/\tau)+\sum_j\exp(V_{j,s}/\tau)}
\]

Outside Option 获得的订单计入：

```text
no_purchase_orders
```

因此所有企业都高价低服务时，总成交需求会下降。

---

# 13. 整数订单分配

所有订单必须是整数。

使用 Largest Remainder：

1. 计算每个候选的理论订单；
2. 取 floor；
3. 剩余订单按小数余数从大到小分配；
4. 余数相同时按 `entity_id` 升序。

这确保：

- 总订单数完全闭合；
- 同输入结果稳定；
- 不依赖字典遍历顺序。

---

# 14. 缺货与一次转售

## 14.1 第一遍

消费者先基于完整选择集选择企业或 Outside Option。

公司因 Capacity 或 Financial Capacity 不能履约的部分成为：

```text
attempted_unfulfilled_orders
```

该指标用于公司 Reputation，因为消费者已经尝试购买但失败。

## 14.2 一次转售

对每个 Segment 的缺货订单：

1. 排除发生缺货的原公司；
2. 只保留有剩余 Capacity 和 Financial Capacity 的公司；
3. 加入 Outside Option；
4. 使用同一 Segment Utility 重新 Softmax；
5. 使用 Largest Remainder 分配一次。

只允许一次转售。

第二遍仍无法满足的订单：

```text
lost_after_stockout_orders
```

不会继续无限循环。

---

# 15. 现金与流动性规则

MVP 不允许：

- 负现金；
- 贷款；
- 公司退出。

## 15.1 固定支出

决策时必须满足：

\[
Ad+Service+Capacity+Resilience+Repair
\le Cash_{start}
\]

否则 Raw Action 非法。

## 15.2 每单贡献

供应冲击可能导致 Unit Cost 高于 Price。

定义：

\[
ContributionPerOrder
=
Price-ActualUnitCost-ExpectedRefundPerOrder
\]

若：

```text
ContributionPerOrder >= 0
```

则现金不限制销量。

若：

```text
ContributionPerOrder < 0
```

则：

\[
FinancialCapacity
=
\left\lfloor
\frac{CashAfterFixedSpend}
{-ContributionPerOrder}
\right\rfloor
\]

最终：

\[
Sales
\le
\min(PotentialDemand, EffectiveCapacity, FinancialCapacity)
\]

因此公司可以亏本销售，但不能把 Cash 结算成负数。

## 15.3 现金耗尽

若 Cash 为 0：

- 仍可选择 0 固定支出动作；
- 若每单贡献为正，可继续销售并恢复现金；
- 若每单贡献为负，Financial Capacity 为 0。

这关闭了“现金为负但公司又不退出”的逻辑空洞。

---

# 16. 产能、折旧与终局

## 16.1 产能生效

Round `t` 的 Capacity Investment：

```text
在 t 轮扣钱
→ 转为 pending_capacity
→ t+1 轮开始时生效
```

## 16.2 折旧

默认每轮 Capacity 和 Capacity Book Value 按配置轻微折旧。

Engineering MVP：

- 有折旧；
- 无 recurring maintenance cost；
- 无 idle capacity cost。

这三个选择必须保持一致，不允许实现者自行添加成本。

## 16.3 `rounds_remaining`

Observation 必须包含：

```text
rounds_remaining
```

## 16.4 最后一轮

当：

```text
rounds_remaining <= 1
```

禁止：

- Capacity Investment；
- Resilience Investment。

因为两者默认下一轮才生效。

## 16.5 Terminal Enterprise Value

Episode 结束时额外计算：

```text
Cash
+ Capacity Book Value × Salvage Rate
+ Awareness Terminal Value
+ Service Terminal Value
+ Reputation Terminal Value
+ Resilience Terminal Value
```

这用于长期策略评测，避免 Agent 只关注最后一轮 Cash。

Terminal Value 不再进入市场交易，只进入 Episode Summary 和 Agent 最终 Utility。

---

# 17. 风险信号与重大事件生命周期

生命周期：

```text
Signal Generated
→ Agent 在目标回合前看到
→ Target Round 前一轮结算时采样是否 Realize
→ Active Event 进入 next_state
→ 按 Duration 持续
→ Recovery
```

## 17.1 预警语义

Risk Signal 包含：

```json
{
  "signal_id": "risk-005",
  "event_type": "extreme_weather",
  "target_round": 6,
  "estimated_probability_ppm": 720000,
  "severity": "high",
  "lead_time_rounds": 1
}
```

所有重大负面事件至少提前 1 轮预警。

Signal 不保证事件发生。

## 17.2 当前韧性与新投资

- `state_t.resilience` 抵御本轮 Active Event；
- `action_t.resilience_budget` 只更新 `state_t+1.resilience`；
- 所以 Agent 必须基于预警提前准备。

## 17.3 Event Distribution

所有：

- 发生概率；
- Severity；
- Duration；
- Multiplier；

由 `market_v4.yaml` 唯一定义。

---

# 18. 公司事故与主动处理

公司事故在 `state_t` 开始时已可见。

Agent 可以：

```text
wait
partial_repair
full_repair
```

## 18.1 Repair Progress

\[
RepairProgress
=
\min(1,
AccumulatedRepairSpend/RepairRequiredCost)
\]

## 18.2 Partial Repair

\[
EffectiveSeverity
=
BaseSeverity
\times
(1-MaxMitigation\times RepairProgress)
\]

## 18.3 Full Repair

如果累计维修达到 Required Cost：

```text
事故在当前销售计算前解决
```

## 18.4 Wait

不花钱，当前轮继续承受事故；结算后自然剩余轮数减 1。

因此公司可以权衡：

```text
立即花钱
vs
部分减轻
vs
承受多个回合损失
```

---

# 19. RNG 跨实现规范

## 19.1 禁止全局随机数

禁止直接调用共享：

```python
random.random()
```

每个随机组件必须独立派生 Seed。

## 19.2 Seed Material 二进制编码

按以下顺序编码：

```text
uint16_be(len(protocol_version_utf8))
protocol_version_utf8
uint64_be(episode_seed)
uint32_be(round)
uint16_be(len(component_name_utf8))
component_name_utf8
uint16_be(len(entity_id_utf8))
entity_id_utf8
uint32_be(draw_index)
```

空 `entity_id` 使用长度 0。

## 19.3 Seed Derivation

```text
digest = SHA-256(seed_material)
sub_seed = unsigned_big_endian(digest[0:8])
```

## 19.4 PRNG

使用：

```text
SplitMix64
```

版本：

```text
rng-splitmix64-v1.0.0
```

所有实现必须使用 64 位无符号溢出语义。

## 19.5 Uniform

```text
u = (next_u64 >> 11) / 2^53
```

得到 `[0,1)`。

## 19.6 Normal Approximation

不使用依赖 `sin/cos/log` 的 Box-Muller。

使用：

```text
z = U1 + U2 + ... + U12 - 6
```

然后按 ppm 量化。

这降低 Python / TypeScript 数学库差异。

## 19.7 Discrete Choice

离散分布权重全部使用整数 ppm。

使用一个 Uniform Draw 与累计整数权重比较。

---

# 20. Canonical JSON 与 State Hash

## 20.1 Canonical JSON

使用：

```text
RFC 8785 JSON Canonicalization Scheme
```

要求：

- 对象 Key 规范排序；
- Enum 使用小写字符串；
- Money / Ratio / Score 均为整数；
- 无 NaN / Infinity；
- 作为 Set 的数组按稳定 ID 排序；
- 排除 `state_hash`、展示字段和时间戳。

## 20.2 Hash

```text
state_hash = "sha256:" + lower_hex(SHA256(canonical_json_utf8))
```

前端可以独立验证 Hash，但不能重新计算市场。

## 20.3 `rng_component_versions`

EpisodeManifest 必须保存：

```json
{
  "rng_component_versions": {
    "demand_noise": "v1",
    "sentiment_noise": "v1",
    "supply_cost_noise": "v1",
    "consumer_utility_noise": "v1",
    "operational_capacity_noise": "v1",
    "risk_signal_generation": "v1",
    "event_realization": "v1",
    "incident_generation": "v1"
  }
}
```

---

# 21. EpisodeManifest

每个正式 Episode 必须先创建 Manifest：

```json
{
  "manifest_version": "episode-manifest-v1.0.0",
  "experiment_id": "exp-001",
  "config_id": "market-v4-default",
  "config_version": "market-v4.0.0",
  "config_sha256": "sha256:...",
  "episode_id": "episode-0001",
  "episode_seed": 42,

  "environment_version": "market-env-v4.0.0",
  "state_schema_version": "market-state-v4.0.0",
  "action_schema_version": "company-action-v4.0.0",
  "event_schema_version": "market-event-v4.0.0",
  "rng_protocol_version": "rng-splitmix64-v1.0.0",
  "hash_protocol_version": "rfc8785-sha256-fixedpoint-v1.0.0",
  "rng_component_versions": {},

  "code_commit": "git-sha",
  "num_agents": 4,
  "max_rounds": 10,
  "information_mode": "perfect",

  "agent_configs": {
    "company_A": {
      "agent_type": "llm",
      "model_provider": "...",
      "model_name": "...",
      "model_version": "...",
      "prompt_version": "...",
      "prompt_sha256": "sha256:...",
      "persona": "aggressive",
      "persona_version": "persona-v1"
    }
  },

  "initial_state": {},
  "initial_state_hash": "sha256:..."
}
```

这份 Manifest 加上 Round Events，必须足以独立重放正式实验。

---

# 22. Round Event

```json
{
  "event_id": "episode-0001:round-04",
  "step_id": "episode-0001:4:3",
  "settled_round": 4,

  "state_before_hash": "sha256:...",
  "state_before": {},
  "observations": {},
  "action_constraints": {},

  "raw_actions": {},
  "validation_results": {
    "company_A": {
      "valid": false,
      "failure_code": "BUDGET_EXCEEDED",
      "retry_count": 1,
      "fallback_used": true
    }
  },
  "final_actions": {},
  "joint_action_hash": "sha256:...",

  "random_draw_summary": {},
  "step_result": {},
  "state_after": {},
  "state_after_hash": "sha256:...",

  "latency_ms": {},
  "token_usage": {},
  "invariant_results": {}
}
```

---

# 23. Persona Utility 规范

Persona 不直接改变 Market Formula。

```text
Persona
→ Planner / Utility
→ Action
→ Environment
→ Result
```

## 23.1 Component Normalization

### Profit Score

\[
ProfitScore
=
clip(RoundProfit/ProfitScale,-1,1)
\]

默认 `ProfitScale = 100000.00`。

### Share Score

\[
ShareScore=MarketShare
\]

范围 `[0,1]`。

### Growth Score

\[
GrowthScore
=
clip((Share_t-Share_{t-1})/0.10,-1,1)
\]

### Stability Score

使用最近 3 个已结算 Round 的利润。

少于 2 个样本：

```text
StabilityScore = 1
```

否则：

\[
StabilityScore
=
1-
\min(StdDev(ProfitWindow)/ProfitScale,1)
\]

### Cash Score

\[
CashScore
=
clip(Cash/(2\times InitialCash),0,1)
\]

### Reputation / Resilience

直接使用 `[0,1]` 值。

## 23.2 Engineering MVP Persona

支持：

- `none`
- `aggressive`
- `conservative`
- `balanced`

权重来自 `market_v4.yaml`。

`cooperative` 只有在 Research MVP 加入 Communication / Welfare 后才启用。

---

# 24. Frontend / Backend 单一计算源

后端：

```text
MarketEnv.step()
```

是唯一市场计算源。

前端只能：

- 获取 Observation；
- 提交 Raw Action；
- 展示 Validation；
- 展示 StepResult；
- 展示 State / Event / Incident；
- 校验 State Hash。

前端禁止：

- 重新计算 Profit；
- 重新计算 Market Share；
- 自行采样随机数；
- 推演 Next State。

这消除 Python 与 TypeScript 两份市场规则漂移。

---

# 25. API 契约

推荐：

```text
POST /episodes
GET  /episodes/{episode_id}/manifest
GET  /episodes/{episode_id}/state
GET  /episodes/{episode_id}/agents/{agent_id}/observation
GET  /episodes/{episode_id}/agents/{agent_id}/action-constraints
POST /episodes/{episode_id}/agents/{agent_id}/actions
POST /episodes/{episode_id}/steps
GET  /episodes/{episode_id}/events
POST /episodes/{episode_id}/replay
```

`POST actions` 和 `POST steps` 都必须支持幂等。

---

# 26. Replay

Replay 输入：

- EpisodeManifest；
- Initial State；
- 每轮 Final Joint Action；
- Config；
- Environment / RNG / Hash Version。

Replay 不调用 LLM。

每轮必须验证：

```text
recomputed_state_after_hash
==
logged_state_after_hash
```

正式验收：

```text
Replay Match Rate = 100%
```

---

# 27. MarketEnv.step() 不变量

每次 Step 后必须满足：

1. `settled_round == state_before.round`；
2. `state_after.round == settled_round + 1`；
3. `state_after.state_version == before + 1`；
4. 每个 Active Company 恰好一个 Final Action；
5. `action_id` 不重复；
6. Joint Action 只执行一次；
7. 所有 Budget 非负且不越界；
8. Fixed Spend 不超过起始 Cash；
9. Cash 不为负；
10. Sales 不超过 Effective / Financial Capacity；
11. 所有 Segment 的初始分配整数闭合；
12. 一次转售后订单闭合；
13. `no_purchase + company_sales + lost_after_stockout == realized_demand`；
14. Awareness / Service / Reputation / Resilience 在 `[0,1]`；
15. 无 NaN / Infinity；
16. 相同 State + Joint Action + Seed + Version 得到相同 Hash；
17. 失败时无部分 State Commit。

---

# 28. 测试计划

## 28.1 Config Test

- YAML 可以加载；
- 权重合计正确；
- Range 合法；
- 事件表字段完整；
- Config Hash 稳定。

## 28.2 Market Formula Test

- 价格敏感度；
- 广告边际递减；
- 服务长期效果；
- Capacity Constraint；
- Outside Option；
- 一次转售；
- Supply Cost；
- Cash Constraint；
- Terminal Value。

## 28.3 Time-order Test

必须专门测试：

- 当前 Resilience 不包含本轮新投资；
- Repair 在销售前生效；
- Capacity 在下一轮生效；
- Risk Signal 对应未来 Round；
- Event 在进入 Active State 后才影响市场；
- 多事件组合顺序不影响结果。

## 28.4 Idempotency Test

- 重复 Action 返回缓存；
- 同 ID 不同 Payload 冲突；
- 重复 Step 返回缓存；
- State Version 过期被拒绝。

## 28.5 RNG / Hash Test Vector

项目仓库必须保存固定 Test Vectors：

```text
seed material
sub_seed
前 10 个 u64
前 10 个 uniform53
normal approximation
canonical JSON
SHA-256
```

Python 和 TypeScript 都运行这些测试，但前端仍不执行市场计算。

## 28.6 Stability

Rule / Mock：

```text
1000 Episodes × 10 Rounds
```

LLM Smoke：

```text
10 Episodes × 10 Rounds
```

---

# 29. 决策价值测试

环境不能只是“复杂”，还必须证明动作有可识别影响。

需要：

- Price vs Margin；
- Advertising vs Cash；
- Service vs Long-term Reputation；
- Capacity vs Future Demand；
- Resilience vs Event Risk；
- Wait vs Partial vs Full Repair；
- Action Effect 大于普通噪声；
- 不同 Seed 有变化、相同 Seed 完全重放。

---

# 30. MVP 验收标准

| 类别 | 要求 |
|---|---|
| Config | 后端只加载 `market_v4.yaml` |
| Environment | 所有公式和时序测试通过 |
| Multi-Agent | 同一 immutable state，同步 Action Lock |
| Final Illegal Action | 0 次进入 Environment |
| Cash | 永不为负 |
| State Invariant | 0 次失败 |
| Idempotency | 全部测试通过 |
| Replay | 100% State Hash 匹配 |
| Rule/Mock Stability | 1000×10 完成 |
| LLM Smoke | 10×10 可完成，非法候选可 fallback |
| Frontend | 不包含市场计算公式 |
| Data | Manifest + Round Events 足以独立 Replay |

---

# 31. Engineering MVP 完成后的下一步

```text
Engineering MVP v4
    ↓
Research MVP A：Persona / Full Information
    ↓
Research MVP B：Imperfect Information / Belief
    ↓
Research MVP C：Communication / Cooperation / Welfare
    ↓
Research MVP D：Game Theory Advisor / Search
    ↓
Self-play / Cross-play / PSRO
    ↓
Policy Dataset / Training
```

---

# 32. 最终定义

Engineering MVP v4 完成的标志是：

> 多个公司 Agent 基于同一个版本化市场状态和动态动作约束独立提交连续数值决策；系统通过 Action Lock、幂等协议和唯一后端 `MarketEnv.step()` 完成市场结算；消费者可以放弃购买或在缺货后转售，公司受现金、产能、事件和事故约束；重大事件有提前预警，事故可以主动维修；所有随机性由版本化 Seed 协议控制，完整 Episode 可以通过 Manifest、Joint Actions 和 State Hash 100% 重放。
