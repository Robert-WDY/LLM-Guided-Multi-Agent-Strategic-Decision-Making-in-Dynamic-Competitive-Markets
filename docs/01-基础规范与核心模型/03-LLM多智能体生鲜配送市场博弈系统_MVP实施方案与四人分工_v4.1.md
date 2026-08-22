# LLM 多智能体生鲜配送市场博弈系统
## Engineering MVP 实施方案、统一接口与四人分工 v4.1

**配套权威文件：**

- `market_v4.yaml`：市场默认参数、事件分布、动作边界的唯一配置来源；
- `02-LLM多智能体生鲜配送市场博弈系统_Engineering_MVP技术规格_v4.0.md`：市场公式、回合时序、RNG、State Hash、幂等和 Replay 的详细规范；
- 本文档：团队分工、统一接口、集成顺序和验收目标。

---

# 1. 项目最终研究方向

完整项目最终研究三个问题：

1. LLM 是否具备 Game-Theoretic Reasoning 能力？
2. LLM Communication 是否会改变博弈结果？
3. LLM 是否能够在不完全信息环境中进行有效推理？

当前 Engineering MVP 不直接回答全部研究问题。

当前阶段先解决：

> 是否已经具备一个正确、稳定、可复现、具有真实决策 Trade-off 的多 Agent 市场基础系统？

只有 Engineering MVP 稳定以后，Communication、Belief、Game Theory Advisor 和 Self-play 的实验结论才可信。

---

# 2. 当前 Engineering MVP 的范围

## 2.1 当前必须实现

当前 MVP 使用：

- 4 个公司 Agent；
- 完全信息；
- 同时行动；
- 20 个回合正式验收；
- 动态市场状态；
- 公司独立状态；
- 价格、广告、服务、产能、韧性和事故维修；
- 市场噪声；
- 灾难风险信号；
- 市场级事件；
- 公司级事故；
- 连续数值动作；
- Preset 转 Numeric Action；
- Rule Agent；
- Mock Agent；
- LLM Agent；
- 简单 Persona；
- Action Validator；
- Action Lock；
- Event Log；
- Replay；
- Evaluator；
- CLI 和基础图表。

当前每个 Agent 每轮主要解决五类经营问题：

```text
pricing
advertising
service
capacity
resilience
```

公司发生事故时增加：

```text
incident repair
```

说明：

> 本文中的 `capacity` 统一定义为“每回合最大履约订单能力”，不是商品库存数量。以后如需库存，应新增独立 `inventory` 字段。

---

## 2.2 当前暂不作为硬验收

以下能力保留接口，但不作为 Engineering MVP 的通过条件：

- 消息真正影响 Agent 决策；
- 公开交流；
- 私聊；
- 合作合同；
- 不完全信息；
- Belief State；
- Opponent Modeling；
- Game Theory Advisor；
- Best Response；
- Regret；
- Self-play；
- PSRO；
- 社会福利优化；
- 8～10 个 LLM Agent 的大规模运行。

这些属于后续 Interaction MVP 或 Research MVP。

---

# 3. MVP 总体架构

```text
                         Experiment Config
                                 |
                                 v
                      Simulation Controller
                                 |
        -------------------------------------------------------
        |                 |                 |                 |
        v                 v                 v                 v
 ObservationBuilder   Agent Runtime   Action Pipeline     Event Logger
        |                 |                 |                 |
        |                 |                 |                 |
        |                 |           Validator + Lock        |
        |                 |                 |                 |
        ------------------|-----------------|------------------
                          |
                          v
                    MarketEnv.step()
                          |
                          v
                       StepResult
                          |
                 ---------------------
                 |                   |
                 v                   v
              Evaluator            View
```

唯一正确的数据流：

```text
MarketEnv 维护真实状态
→ ObservationBuilder 构造 Agent 观察
→ Agent 生成 Raw Action
→ Validator 生成 Final Action
→ Action Lock 生成 Joint Action
→ MarketEnv 一次性更新
→ Logger 记录
→ Evaluator 计算
→ View 展示
```

禁止：

- Agent 直接修改 MarketState；
- Agent 直接调用 `MarketEnv.step()`；
- 前端自行计算利润和市场份额；
- 多 Agent 逐个更新市场；
- 不同模块复制一套市场公式。

---

# 4. 规范来源与变更规则

## 4.1 规范优先级

```text
1. market_v4.yaml
2. Engineering MVP 技术规格 v4.0
3. 本实施与分工文档 v4.1
4. 完整项目报告
5. 历史版本文档
```

如果文档冲突，以优先级更高的规范为准。

---

## 4.2 接口冻结规则

开始并行开发前，必须冻结以下 Schema：

```text
MarketState
CompanyState
AgentObservation
ActionConstraints
PlannerOutput
RawAgentAction
FinalAgentAction
JointAction
StepResult
EpisodeManifest
RoundEvent
EvaluationSummary
```

Schema 修改必须：

1. 修改版本号；
2. 修改对应 JSON Schema / Typed Model；
3. 增加迁移说明；
4. 更新 Contract Test；
5. 由接口与集成 Owner 合并。

---

# 5. 统一接口

# 5.1 MarketEnv 接口

```python
class MarketEnv:
    def reset(
        self,
        manifest: "EpisodeManifest"
    ) -> "MarketState":
        ...

    def get_action_constraints(
        self,
        state: "MarketState",
        agent_id: str
    ) -> "ActionConstraints":
        ...

    def step(
        self,
        state: "MarketState",
        joint_action: "JointAction",
        step_id: str
    ) -> "StepResult":
        ...

    def validate_invariants(
        self,
        state: "MarketState"
    ) -> list["InvariantViolation"]:
        ...
```

约束：

- `MarketEnv` 不调用 LLM；
- `MarketEnv` 不生成 Agent 计划；
- `step()` 每回合只调用一次；
- `step()` 必须幂等；
- 相同 State、Joint Action、Seed 和版本必须得到相同结果。

---

# 5.2 ObservationBuilder 接口

```python
class ObservationBuilder:
    def build(
        self,
        state: "MarketState",
        agent_id: str,
        information_mode: str
    ) -> "AgentObservation":
        ...
```

Engineering MVP：

```text
information_mode = perfect_information
```

后续不完全信息不修改 Agent 主接口，只替换 ObservationBuilder。

---

# 5.3 Agent 接口

```python
class Agent:
    def observe(
        self,
        observation: "AgentObservation"
    ) -> None:
        ...

    def plan(
        self,
        observation: "AgentObservation",
        constraints: "ActionConstraints"
    ) -> "PlannerOutput":
        ...

    def generate_action(
        self,
        plan: "PlannerOutput",
        constraints: "ActionConstraints"
    ) -> "RawAgentAction":
        ...

    def analyze_result(
        self,
        result: "AgentRoundResult"
    ) -> "ResultAnalysis":
        ...
```

所有 Agent 类型必须实现同一接口：

- RuleAgent；
- MockAgent；
- LLMAgent。

---

# 5.4 Agent 工具接口

Engineering MVP 中 Agent 只允许访问：

```text
get_current_observation
get_action_constraints
get_recent_own_history
submit_raw_action
```

禁止 Agent 工具：

- 直接读取未授权私有状态；
- 直接修改环境；
- 直接调用 `MarketEnv.step()`；
- 提前读取尚未实现的随机事件；
- 绕过 Validator 提交 Final Action。

---

# 5.5 PlannerOutput

```json
{
  "agent_id": "company_A",
  "round": 6,
  "state_version": 5,

  "objective": "提高长期公司收益",

  "situation_summary": "当前产能利用率较高，下一轮存在极端天气风险，竞争者B正在降价。",

  "key_factors": [
    "cash_balance",
    "capacity_utilization",
    "competitor_price",
    "risk_signal",
    "active_incident"
  ],

  "constraints_considered": [
    "固定支出不得超过现金",
    "销量不能超过有效产能",
    "产能投资下一轮生效"
  ],

  "strategy_summary": "适度提价，减少无效广告，增加产能与韧性投入。",

  "expected_effects": {
    "short_term_profit": "decrease",
    "future_capacity": "increase",
    "disaster_loss": "decrease"
  }
}
```

Planner 只负责形成当前回合战略，不直接执行动作。

---

# 5.6 RawAgentAction

```json
{
  "action_id": "episode-001-r06-company-A-attempt-1",
  "episode_id": "episode-001",
  "agent_id": "company_A",
  "round": 6,
  "state_version": 5,

  "price_cents": 9600,
  "advertising_budget_cents": 1200000,
  "service_budget_cents": 1400000,
  "capacity_investment_cents": 1800000,
  "resilience_budget_cents": 800000,

  "incident_response": {
    "mode": "partial_repair",
    "repair_budget_cents": 1000000
  },

  "strategy_summary": "在控制现金支出的同时为下一轮扩产并降低灾害风险。"
}
```

---

# 5.7 ActionConstraints

连续动作空间不使用旧式 `legal_actions()` 枚举全部动作。

统一使用：

```json
{
  "agent_id": "company_A",
  "state_version": 5,

  "bounds": {
    "price_cents": {
      "min": 7500,
      "max": 13000
    },
    "advertising_budget_cents": {
      "min": 0,
      "max": 5000000
    },
    "service_budget_cents": {
      "min": 0,
      "max": 4000000
    },
    "capacity_investment_cents": {
      "min": 0,
      "max": 5000000
    },
    "resilience_budget_cents": {
      "min": 0,
      "max": 2000000
    },
    "repair_budget_cents": {
      "min": 0,
      "max": 8000000
    }
  },

  "available_cash_cents": 25000000,
  "active_incident": null,
  "delayed_effects": {
    "capacity_investment": 1,
    "resilience_investment": 1
  }
}
```

---

# 5.8 Individual Action Validator

```python
class IndividualActionValidator:
    def validate(
        self,
        raw_action: "RawAgentAction",
        constraints: "ActionConstraints"
    ) -> "ActionValidationResult":
        ...

    def repair_or_fallback(
        self,
        raw_action: "RawAgentAction",
        validation: "ActionValidationResult",
        previous_action: "FinalAgentAction | None"
    ) -> "FinalAgentAction":
        ...
```

负责：

- JSON / 类型；
- 必填字段；
- 数值范围；
- Cash 约束；
- Round；
- State Version；
- Agent ID；
- Incident Response 合法性；
- Retry；
- Fallback。

最终要求：

```text
Final Invalid Action Rate = 0
```

---

# 5.9 Action Lock

```python
class ActionLock:
    def lock(
        self,
        state: "MarketState",
        final_actions: dict[str, "FinalAgentAction"]
    ) -> "JointAction":
        ...
```

检查：

- 每个 active Agent 恰好一个动作；
- 所有动作对应同一 Episode；
- 所有动作对应同一 Round；
- 所有动作对应同一 State Version；
- Action ID 不重复；
- Final Action 已通过校验；
- Lock 后不可修改。

---

# 5.10 SimulationController

```python
class SimulationController:
    def run_round(
        self,
        state: "MarketState"
    ) -> "StepResult":
        ...

    def run_episode(
        self,
        manifest: "EpisodeManifest"
    ) -> "EpisodeResult":
        ...
```

每轮固定流程：

```text
freeze state_t
→ build observations
→ collect plans
→ collect raw actions
→ validate
→ action lock
→ one joint env.step()
→ log
→ evaluate
→ feedback
```

---

# 5.11 EventLogger

```python
class EventLogger:
    def write_manifest(
        self,
        manifest: "EpisodeManifest"
    ) -> None:
        ...

    def write_round_event(
        self,
        event: "RoundEvent"
    ) -> None:
        ...

    def write_episode_summary(
        self,
        summary: "EvaluationSummary"
    ) -> None:
        ...
```

---

# 5.12 Evaluator

```python
class Evaluator:
    def evaluate_round(
        self,
        event: "RoundEvent"
    ) -> "RoundMetrics":
        ...

    def evaluate_episode(
        self,
        manifest: "EpisodeManifest",
        events: list["RoundEvent"]
    ) -> "EvaluationSummary":
        ...

    def compare_conditions(
        self,
        experiment_ids: list[str]
    ) -> "ExperimentComparison":
        ...
```

---

# 5.13 DecisionAdvisor 插件接口

为了后续加入 Game Theory，不直接修改 Agent Loop。

```python
class DecisionAdvisor:
    def advise(
        self,
        observation: "AgentObservation",
        plan: "PlannerOutput",
        constraints: "ActionConstraints"
    ) -> "AdvisorOutput":
        ...
```

Engineering MVP 使用：

```text
NoOpDecisionAdvisor
```

Research MVP 可替换为：

```text
GameTheoryAdvisor
```

---

# 5.14 Communication 插件接口

```python
class CommunicationModule:
    def run_phase(
        self,
        state: "MarketState",
        agents: list["Agent"]
    ) -> "CommunicationResult":
        ...
```

Engineering MVP 使用：

```text
NoOpCommunicationModule
```

后续可替换：

- PublicMessageBus；
- PrivateMessageBus；
- NegotiationModule。

---

# 6. 四人主体分工

# 6.1 成员一：市场环境

## 核心目标

构建唯一、正确、可复现、具有真实决策 Trade-off 的生鲜配送市场。

## 负责范围

### 市场状态

负责：

- MarketState；
- CompanyState；
- Market Event；
- Risk Signal；
- Company Incident；
- Terminal State。

### 市场动作影响

负责：

- Pricing；
- Advertising；
- Service；
- Capacity；
- Resilience；
- Incident Repair。

### 市场计算

负责：

- 消费者分群选择；
- Outside Option；
- Demand；
- Stockout；
- 一次转售；
- Sales；
- Revenue；
- Cost；
- Profit；
- Cash；
- Awareness；
- Service Quality；
- Reputation；
- Capacity；
- Resilience。

### 随机性与灾难

负责：

- Seed 派生；
- 需求波动；
- 消费者偏好波动；
- 运营波动；
- Risk Signal；
- Market Event；
- Company Incident；
- 相同 Seed Replay。

### 工程接口

负责实现：

```text
MarketEnv.reset()
MarketEnv.get_action_constraints()
MarketEnv.step()
MarketEnv.validate_invariants()
```

## 交付物

```text
configs/market_v4.yaml
environment/state.py
environment/market_env.py
environment/transition.py
environment/events.py
environment/incidents.py
environment/rng.py
environment/invariants.py
tests/test_market_formula.py
tests/test_rng_replay.py
tests/test_event_lifecycle.py
```

## 独立验收目标

### Action Sensitivity

固定其他参数：

```text
price下降
→ 平均market_share上升
```

### Trade-off

低价必须体现：

```text
share上升
+
unit margin下降
```

### Capacity Constraint

必须：

```text
sales <= effective_capacity
```

### Long-term Effect

持续高服务投入后：

```text
service_quality上升
→ reputation逐步上升
```

### Stochasticity与复现

必须同时满足：

```text
相同State + 相同Joint Action + 相同Seed
→ 相同结果

不同Seed
→ 允许不同结果
```

### Agency > Randomness

普通噪声不能压倒动作效果。

### Disaster Strategy

灾难发生时：

```text
高resilience公司平均损失
<
低resilience公司平均损失
```

### Incident Response

同一事故下：

```text
Wait
Partial Repair
Full Repair
```

必须产生不同维修成本、恢复时间和累计损失。

### 稳定性

```text
1000 Episodes
×
20 Rounds
×
4 RuleAgents
```

要求：

- State invariant violation = 0；
- Replay mismatch = 0；
- Environment crash = 0。

---

# 6.2 成员二：单 Agent 内部流程

## 核心目标

实现一个 Agent 从获取市场信息到分析结果的完整决策闭环。

## 负责范围

### Agent Runtime

实现：

```text
Observe
→ Analyze
→ Plan
→ Generate Raw Action
→ Validate / Repair
→ Receive Result
→ Analyze Result
```

### Agent 类型

实现：

- RuleAgent；
- MockAgent；
- LLMAgent。

### Planner

Planner 必须考虑：

- 公司现金；
- 当前利润；
- 市场份额；
- 产能利用率；
- 服务和声誉；
- Risk Signal；
- Active Incident；
- 竞争者状态；
- Action Constraints；
- Persona。

### Agent 合法性

负责 Individual Action Validator：

- 参数范围；
- Cash；
- Round；
- State Version；
- Incident Repair；
- Retry；
- Fallback。

### Persona

Engineering MVP 支持：

- aggressive；
- conservative；
- balanced；
- none。

Persona 只通过：

```text
Persona
→ Goal / Utility / Prompt
→ Action
```

影响结果。

禁止 Environment 根据 Persona 直接给奖励或市场份额加成。

### 可解释性

保存结构化：

- Situation Summary；
- Objective；
- Key Factors；
- Constraints；
- Strategy Summary；
- Expected Effects；
- Result Analysis。

不要求保存无限自由文本推理。

## 交付物

```text
agents/base_agent.py
agents/rule_agent.py
agents/mock_agent.py
agents/llm_agent.py
agents/planner.py
agents/result_analyzer.py
validation/action_validator.py
validation/fallback.py
schemas/agent_observation.schema.json
schemas/planner_output.schema.json
schemas/raw_agent_action.schema.json
tests/test_agent_contract.py
tests/test_action_validator.py
tests/test_persona_injection.py
```

## 独立验收目标

### 单 Agent 闭环

使用：

```text
1个被测Agent
+
3个固定RuleAgent对手
```

连续运行：

```text
20 Rounds
```

被测 Agent 每轮均完成：

```text
Observation
→ Planner
→ Raw Action
→ Final Action
→ Result Analysis
```

### RuleAgent

相同 Observation 和配置：

```text
输出完全一致
```

### MockAgent

可以稳定注入：

- 非 JSON；
- 缺字段；
- 超范围；
- 过期 State Version；
- 超预算；
- Timeout。

系统能够处理这些错误。

### LLMAgent Smoke Test

至少：

```text
5 Episodes
×
20 Rounds
```

要求：

- 可以端到端运行；
- Raw Output 被完整记录；
- 非法候选被检测；
- 非法候选不能进入 MarketEnv；
- Fallback 后 Episode 可以继续。

### Persona

固定：

```text
相同State
相同Seed
相同Model
相同Prompt主体
```

只改变 Persona。

系统必须能够比较：

- Price；
- Advertising；
- Service；
- Capacity；
- Resilience；
- Repair。

研究结果可以是“差异不显著”，但数据链路必须完整。

### 可解释性

每个动作必须能追溯到：

```text
Observation
PlannerOutput
RawAction
Validation
FinalAction
ResultAnalysis
```

---

# 6.3 成员三：多 Agent 交互与系统集成

## 核心目标

让多个 Agent 基于同一个状态稳定同时决策，并保证整个系统可插拔、可扩展、无信息泄漏和无时序错误。

该成员同时担任：

> 接口与集成 Owner。

## 负责范围

### Simulation Controller

负责：

- Episode 生命周期；
- Round 调度；
- Observation 分发；
- Agent 调用；
- Action 收集；
- Action Lock；
- Joint Action；
- `MarketEnv.step()`；
- Feedback 分发。

### 同时行动协议

必须严格实现：

```text
所有Agent读取同一个immutable state_t
→ 独立生成动作
→ Validator
→ Action Lock
→ Joint Action
→ 每轮一次env.step()
→ state_t+1
```

### 扩展性

MVP 硬验收：

```text
4 Agents
```

扩展测试：

```text
8 Agents RuleAgent
```

10 Agent 暂不作为硬验收。

### 可插拔模块

Controller 必须支持：

- Agent 类型可替换；
- ObservationBuilder 可替换；
- DecisionAdvisor 可替换；
- CommunicationModule 可替换；
- Evaluator 可替换；
- View 可替换。

### 信息模式

当前：

```text
perfect_information
```

预留：

```text
imperfect_information
noisy_information
```

### Communication

Engineering MVP 使用 NoOp。

后续由该成员扩展：

- Public Message；
- Private Message；
- Visibility；
- Communication Close；
- Communication 影响 Agent 决策。

### 统一接口

负责维护：

```text
schemas/
API版本
Contract Tests
Integration Branch
```

任何跨模块 Schema 变更由该成员审核合并。

## 交付物

```text
controller/simulation_controller.py
controller/episode_runner.py
controller/action_lock.py
observation/observation_builder.py
communication/base.py
communication/noop.py
advisors/base.py
advisors/noop.py
schemas/joint_action.schema.json
schemas/step_result.schema.json
tests/test_action_lock.py
tests/test_simultaneous_action.py
tests/test_observation_visibility.py
tests/test_full_episode_integration.py
```

## 独立验收目标

### 同时行动

验证：

- 所有 Agent Observation 的 State Version 相同；
- 没有 Agent 看到其他 Agent 当前轮 Final Action；
- 每轮只调用一次 `MarketEnv.step()`；
- Joint Action 中每个 Agent 恰好一个动作。

### 四 Agent 正式验收

```text
4 Agents
×
20 Rounds
```

要求完整结束。

### 集成稳定性

Rule / Mock：

```text
100 Episodes
×
20 Rounds
```

要求：

- Episode completion = 100%；
- Missing Action = 0；
- Duplicate Action = 0；
- State Version Conflict = 0；
- Partial State Update = 0。

### 扩展测试

```text
8 RuleAgents
×
20 Rounds
```

能够运行，且不修改 MarketEnv、Agent 和 Evaluator 核心接口。

### Visibility

完全信息模式中字段符合定义。

后续切换不完全信息时：

- 不需要修改 Agent Interface；
- 只替换 ObservationBuilder / Visibility Policy。

### 接口冻结

所有 Contract Tests 在合并前必须通过。

---

# 6.4 成员四：测试、评估与博弈论分析

## 核心目标

建立可以判断系统是否正确、Agent 是否有效，以及后续 Game Theory 是否有帮助的实验与评估体系。

Engineering MVP 阶段优先完成：

> Evaluation、Experiment、Data、Replay 和 View。

Game Theory Advisor 在 Engineering MVP 通过后接入。

## 负责范围

### Experiment Runner

支持：

- Config；
- Seed；
- Agent 组合；
- Persona；
- Model；
- Prompt Version；
- 回合数；
- 多 Episode 批量运行。

### 数据记录

负责：

- EpisodeManifest；
- RoundEvent；
- EvaluationSummary；
- ExperimentComparison；
- 数据导出。

### Replay

根据：

```text
Initial State
+
Final Joint Actions
+
Seed
+
Environment Version
+
Config Version
```

重放 Episode。

### 工程评估

负责：

- Completion；
- Error；
- Invalid Raw Action；
- Fallback；
- Invariant；
- Replay；
- Latency；
- Token；
- Cost。

### Agent评估

负责：

- Profit；
- Cumulative Profit；
- Market Share；
- Profit Volatility；
- Action Distribution；
- Persona Difference；
- Risk Response；
- Repair Decision。

### 市场评估

负责：

- Demand；
- Outside Option；
- Price；
- Sales；
- Stockout；
- Capacity Utilization；
- Event Loss；
- Incident Loss。

### View

提供：

- CLI；
- Profit Chart；
- Market Share Chart；
- Cash Chart；
- Capacity Chart；
- Reputation Chart；
- Risk / Event Timeline；
- Incident / Repair Timeline。

### Game Theory 插件

后续实现：

```text
GameTheoryAdvisor
```

但不直接改变 Controller 接口。

研究比较：

```text
Persona-only
vs
Persona + Game Theory Advisor
```

### 防止对手过拟合

后续设计：

- Cross-play；
- 不同 Persona；
- 不同 Rule Policy；
- 不同 Model；
- 不同市场参数；
- Holdout Opponent；
- Self-play Population。

## 交付物

```text
experiments/runner.py
experiments/configs/
data/episode_manifest.py
data/round_event.py
data/event_logger.py
data/replay.py
evaluation/evaluator.py
evaluation/engineering_metrics.py
evaluation/agent_metrics.py
evaluation/market_metrics.py
evaluation/comparison.py
view/cli.py
view/charts.py
view/report.py
game_theory/advisor.py
tests/test_event_log_completeness.py
tests/test_replay.py
tests/test_metrics.py
```

## 独立验收目标

### EpisodeManifest完整

每局至少记录：

- Experiment ID；
- Config ID；
- Seed；
- Environment Version；
- RNG Version；
- Agent Type；
- Persona；
- Model；
- Model Version；
- Prompt Version；
- Initial State Hash。

### RoundEvent完整

每轮记录：

- State Before；
- Observation；
- PlannerOutput；
- Raw Action；
- Validation Error；
- Retry Count；
- Fallback；
- Final Action；
- Joint Action；
- RNG Component Summary；
- Event；
- Incident；
- StepResult；
- State After；
- Metrics。

### Replay

```text
Replay Match Rate = 100%
```

### 错误归因

至少能够区分：

```text
Model Error
Validation Error
Controller Error
Environment Error
Random Event
Experiment Config Error
```

### Evaluation

至少生成：

- 单 Episode Summary；
- 多 Episode Aggregation；
- Persona Comparison；
- Action Sensitivity；
- Disaster Strategy；
- Incident Response；
- Engineering Stability Report。

### 前端/展示

View 不能自行计算市场结果。

所有图表只读取：

```text
StepResult
RoundEvent
EvaluationSummary
```

### 博弈论预留验收

Engineering MVP 中：

```text
NoOpAdvisor
```

可以被：

```text
GameTheoryAdvisor
```

替换，而无需修改 Agent、Controller 和 MarketEnv 公共接口。

---

# 7. 跨模块职责矩阵

| 产物 | 主负责人 | 协作人 | 最终合并 |
|---|---|---|---|
| `market_v4.yaml` | 市场环境 | 测试评估 | 接口与集成 Owner |
| MarketState / CompanyState | 市场环境 | 多 Agent | 接口与集成 Owner |
| Observation Schema | 多 Agent | 单 Agent、市场环境 | 接口与集成 Owner |
| Planner / Agent Action | 单 Agent | 多 Agent | 接口与集成 Owner |
| Individual Validator | 单 Agent | 市场环境 | 接口与集成 Owner |
| Action Lock / Joint Action | 多 Agent | 单 Agent | 接口与集成 Owner |
| StepResult | 市场环境 | 多 Agent、测试评估 | 接口与集成 Owner |
| EpisodeManifest | 测试评估 | 全员 | 接口与集成 Owner |
| RoundEvent | 测试评估 | 全员 | 接口与集成 Owner |
| Replay | 测试评估 | 市场环境 | 接口与集成 Owner |
| CLI / Charts | 测试评估 | 多 Agent | 接口与集成 Owner |
| Game Theory Advisor | 测试评估 | 单 Agent | 接口与集成 Owner |
| Communication | 多 Agent | 单 Agent、测试评估 | 接口与集成 Owner |

---

# 8. MVP 内部开发 Gate

# Gate 0：接口冻结

## 实现

- `market_v4.yaml`；
- 统一单位；
- 所有核心 Schema；
- Interface Stub；
- Contract Test；
- 示例 State / Action / Event。

## 效果

四个人可以并行开发，不会因为字段和接口不一致反复返工。

## 验收

- 所有 Schema 可加载；
- 所有 Stub 可导入；
- 示例数据通过 Schema；
- CI Contract Test 通过。

---

# Gate 1：市场环境可独立运行

## 实现

- MarketEnv；
- Event；
- Incident；
- RNG；
- State Hash；
- Rule Action；
- Invariant。

## 效果

不接入 LLM，也能用固定 Joint Action 跑完整市场。

## 验收

```text
1000 Episodes × 20 Rounds
```

无环境错误，Replay 100%。

---

# Gate 2：单 Agent闭环

## 实现

- Agent Interface；
- RuleAgent；
- MockAgent；
- LLMAgent；
- Planner；
- Validator；
- Fallback；
- Result Analysis。

## 效果

一个被测 Agent 可以基于公司状态做出完整经营决策。

## 验收

```text
1个被测Agent + 3个RuleAgent
×
20 Rounds
```

完整运行。

---

# Gate 3：四 Agent 同时运行

## 实现

- Controller；
- ObservationBuilder；
- Action Lock；
- Joint Action；
- 同时行动；
- NoOp Communication；
- NoOp Advisor。

## 效果

4 个公司在同一个市场中稳定竞争，每个动作共同影响市场。

## 验收

```text
4 Agents × 20 Rounds
```

每轮一次 Joint Step，无状态泄漏和时序错误。

---

# Gate 4：数据、Replay和评估

## 实现

- EpisodeManifest；
- RoundEvent；
- Replay；
- Evaluator；
- CLI；
- 图表；
- Stability Report。

## 效果

任何一次结果都可以解释：

```text
Agent看到了什么
→ 为什么这样计划
→ 原始动作是什么
→ 是否被修复
→ 市场发生了什么
→ 最终结果如何
```

## 验收

- Event完整率 100%；
- Replay Match 100%；
- 必要图表全部生成；
- 错误可以归因。

---

# Gate 5：LLM与Persona Smoke Test

## 实现

- 真实 Model；
- Prompt Version；
- Persona；
- Numeric Action；
- Token / Cost；
- Result Analysis。

## 效果

LLM Agent 可以真实参与市场，而不只是 Rule/Mock 测试。

## 验收

```text
5 Episodes
×
4 LLMAgents
×
20 Rounds
```

在 API 可用情况下完成；非法候选可被处理，环境最终非法动作数为 0。

---

# 9. Engineering MVP 整体验收表

| 类别 | 指标 | 目标 |
|---|---|---:|
| 市场 | Action Sensitivity | 趋势符合定义 |
| 市场 | Price Trade-off | Share↑且Margin↓ |
| 市场 | Sales ≤ Capacity | 100% |
| 市场 | Service长期影响 | 可测 |
| 市场 | Same Seed Replay | 100%一致 |
| 市场 | Agency > Normal Noise | 通过效应量测试 |
| 灾难 | 高韧性平均损失更低 | 通过配对Seed测试 |
| 事故 | Wait/Partial/Full结果不同 | 通过 |
| Agent | Final Invalid Action | 0 |
| Agent | Planner/Action/Feedback链路 | 100%完整 |
| Multi-Agent | 同一State Version | 100% |
| Multi-Agent | 每轮Joint Step次数 | 1 |
| Multi-Agent | 4 Agents × 20 Rounds | 完成 |
| Controller | Duplicate Action | 0 |
| Controller | Partial Update | 0 |
| Data | Manifest完整率 | 100% |
| Data | RoundEvent完整率 | 100% |
| Replay | State Hash Match | 100% |
| View | 前端独立市场计算 | 0 |
| 稳定性 | Rule 1000×20 | 无崩溃/不变量错误 |
| LLM | 5 Episodes×20 | Smoke Test完成 |

---

# 10. 测试分层

## 10.1 Unit Test

测试：

- 公式；
- RNG；
- Hash；
- Action Mapping；
- Validator；
- Event；
- Incident；
- Utility；
- Metrics。

---

## 10.2 Contract Test

测试：

- Schema；
- Interface；
- Version；
- Example Payload；
- 前后模块输入输出兼容。

---

## 10.3 Integration Test

测试：

```text
Agent
→ Validator
→ Controller
→ MarketEnv
→ Logger
→ Evaluator
```

---

## 10.4 Stability Test

Rule / Mock 为主：

```text
1000 Episodes × 20 Rounds
```

LLM 不承担大规模工程稳定性测试。

---

## 10.5 LLM Smoke Test

少量真实模型：

```text
5 Episodes × 20 Rounds
```

测试接口、Prompt、Validator和Fallback。

---

## 10.6 Research Test

Engineering MVP 之后再进行：

- Persona；
- Communication；
- Imperfect Information；
- Game Theory；
- Cross-play；
- Social Welfare。

---

# 11. EpisodeManifest 最低字段

```json
{
  "experiment_id": "engineering-mvp-001",
  "config_id": "market-v4-default",
  "episode_id": "episode-0001",

  "episode_seed": 42,

  "environment_version": "market-env-v4.0.0",
  "config_version": "market-v4.0.0",
  "rng_protocol_version": "rng-splitmix64-v1.0.0",
  "hash_protocol_version": "rfc8785-sha256-fixedpoint-v1.0.0",

  "information_mode": "perfect_information",
  "communication_module": "noop",
  "decision_advisor": "noop",

  "num_agents": 4,
  "num_rounds": 20,

  "agents": {
    "company_A": {
      "agent_type": "llm",
      "persona": "aggressive",
      "model_name": "model-name",
      "model_version": "model-version",
      "prompt_version": "prompt-v1"
    }
  },

  "initial_state": {},
  "initial_state_hash": "sha256:..."
}
```

---

# 12. RoundEvent 最低字段

```json
{
  "event_id": "episode-0001-round-06",
  "episode_id": "episode-0001",
  "settled_round": 6,

  "state_before": {},
  "state_before_hash": "sha256:...",

  "observations": {},
  "planner_outputs": {},
  "raw_actions": {},

  "validation_results": {
    "company_A": {
      "valid": false,
      "error_type": "budget_exceeded",
      "retry_count": 1,
      "fallback_used": true
    }
  },

  "final_actions": {},
  "joint_action": {},
  "joint_action_hash": "sha256:...",

  "rng_component_summary": {},
  "market_events": [],
  "company_incidents": {},

  "step_result": {},

  "state_after": {},
  "state_after_hash": "sha256:...",

  "round_metrics": {},
  "latency": {},
  "token_usage": {}
}
```

---

# 13. 代码目录与主责任

```text
project/
├── configs/                  # 成员一主负责
├── environment/              # 成员一
├── agents/                   # 成员二
├── validation/               # 成员二
├── observation/              # 成员三
├── controller/               # 成员三
├── communication/            # 成员三
├── advisors/                 # 成员四后续，成员三维护接口
├── data/                     # 成员四
├── experiments/              # 成员四
├── evaluation/               # 成员四
├── view/                     # 成员四
├── schemas/                  # 全员提交，成员三发布版本
└── tests/                    # 各自模块测试 + 共同集成测试
```

---

# 14. 协作规则

## 14.1 单一计算源

只有后端 `MarketEnv.step()` 计算：

- Demand；
- Market Share；
- Sales；
- Profit；
- State Update。

前端、Agent和Evaluator不重复实现市场公式。

---

## 14.2 版本化

每次变更必须更新对应版本：

```text
Environment Version
Config Version
State Schema Version
Action Schema Version
Prompt Version
Evaluator Version
```

---

## 14.3 Definition of Done

任何模块完成必须满足：

- 接口实现；
- Unit Test；
- Contract Test；
- 错误处理；
- Event字段；
- 文档；
- 无硬编码配置副本；
- CI通过。

---

## 14.4 集成节奏

建议：

```text
每天：
各模块运行自己的Unit Test

每2～3天：
合并Integration Branch

每个Gate结束：
固定版本 + 运行完整验收
```

---

# 15. 当前MVP不应该做的事情

为了避免范围失控，当前阶段不应该：

- 一开始实现完整私聊和谈判；
- 一开始同时做完全信息和不完全信息；
- 一开始实现 PSRO；
- 用前端复制市场算法；
- 让 LLM 直接修改 State；
- 用 LLM 进行1000局稳定性测试；
- 把8～10个LLM Agent作为硬验收；
- 在MarketEnv中根据Persona直接加收益；
- 在没有Event Log时开始正式研究实验。

---

# 16. Engineering MVP 最终完成定义

Engineering MVP 完成的标志是：

> 4 个公司 Agent 可以基于同一个不可变市场状态，在完全信息条件下独立完成观察、规划和数值动作生成；所有动作经过独立校验与 Action Lock 后被 MarketEnv 一次性执行；市场能够体现价格、广告、服务、产能、韧性和事故维修的短期及长期影响；随机事件可由 Seed 完全重放；系统可以稳定运行 20 回合，并通过完整 Event Log、Replay 和 Evaluator 解释每一个结果。

Engineering MVP 通过后，团队才进入：

```text
Communication
→ Imperfect Information
→ Belief
→ Game Theory
→ Self-play
→ Policy Training
```
