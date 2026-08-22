# Agent Runtime 与多 Agent 协调器

这一阶段采用固定工作流，不使用自由 ReAct：

```text
Observation → DecisionContext → AgentDecision → Intent
            → Action Lock / Rule fallback → MarketEnv.step（一次）
            → ResultAnalysis → EpisodeMemory → RoundEvent
```

Agent 只决定“希望采用什么经营策略”。受信任的 RoundCoordinator 决定“何时收集决策、哪些意图进入本轮、何时统一结算”。任何 Agent 或模型都不能直接调用 `MarketEnv.step()`。

## 1. 代码边界

| 模块 | 职责 |
| --- | --- |
| `agents/contracts.py` | AgentDecision、DecisionContext、ResultAnalysis 等版本化 Pydantic 契约 |
| `agents/observation.py` | 版本化 `perfect` / `public` Visibility Policy 与唯一 TrueState→View 投影 |
| `information/` | Observation Snapshot、View Hash 与 Information Replay |
| `agents/runtime.py` | 独立 Communicate 与 Decide/Validate 流程、超时和错误归一化 |
| `agents/memory.py` | 最近 5 轮、滚动统计、关键事件，不保存完整对话 |
| `agents/result_analyzer.py` | 预期与结算结果的确定性比较 |
| `agents/market_regime.py` | 不改变市场公式的竞争、需求、产能、成本、集中度和风险分类 |
| `model_clients/base.py` | 模型提供方的最小异步 Protocol |
| `model_clients/mock.py` | 无外部模型依赖的透明测试 Agent |
| `orchestration/coordinator.py` | 冻结同一状态、并发决策、意图收集、fallback、单次结算 |
| `orchestration/round_event.py` | 决策链、规则修正、最终动作与结果分析 JSONL |
| `orchestration/clients.py` | 8011 Agent 平面与 8010 私有控制平面的异步 HTTP 客户端 |

## 2. 单 Agent 的正确运行方式

“单 Agent”仍然创建四家公司。A 使用正在测试的 Agent，B/C/D 不注册 Runtime，由 Controller 的 `rule` fallback 补齐：

```python
from game_theory_agent.agents import AgentRuntime
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.orchestration import (
    HttpAgentGatewayClient,
    HttpControllerClient,
    JsonlRoundEventLogger,
    RoundCoordinator,
)

runtime = AgentRuntime(
    agent_id="planner-A",
    company_id="company_A",
    model_client=MockModelClient(),
)
coordinator = RoundCoordinator(
    controller=HttpControllerClient("<controller-token>"),
    gateway=HttpAgentGatewayClient(),
    runtimes={"company_A": runtime},
    event_logger=JsonlRoundEventLogger("runs/episode-001/agent-rounds.jsonl"),
)

# Episode 必须先由 8010 的 POST /api/episodes 创建。
rounds = await coordinator.run_episode("episode-001")
```

未注册 Runtime 的公司不是“缺失错误”，而是显式使用稳定规则对手。这样可以先验证一个智能 Agent 是否真的能在竞争环境中工作。

## 3. 多 Agent

为更多公司注册 Runtime 即可，协调协议不变：

```python
runtimes = {
    company_id: AgentRuntime(
        agent_id=f"planner-{company_id}",
        company_id=company_id,
        model_client=make_model_client(company_id),
    )
    for company_id in ("company_A", "company_B", "company_C", "company_D")
}
```

RoundCoordinator 先从 Controller 读取一次权威状态，再并发读取观察，并强制所有观察的 `round`、`state_version`、`state_hash` 与冻结状态一致。`communication_mode=off` 时保持原路径；开启通信时，所有 Runtime 先并发生成一次 Cheap Talk，Controller 关闭通信并生成公司级可见视图，协调器重新读取关闭后的 Observation，随后才并发规划经营动作。Agent 仍互相看不到本轮最终动作，且消息不会直接修改市场。完整状态机和验收见 `docs/05-交互合作与综合实验/01-phase2-interaction-mvp-design.md`。

## 4. 失败策略

| 情况 | 当前处理 |
| --- | --- |
| 模型超时 | 该公司本轮 Rule fallback |
| 模型异常或 Schema 非法 | 记录错误，该公司 Rule fallback |
| Intent 提交失败 | 记录错误，该公司 Rule fallback |
| Intent 返回 409 | 视为冻结状态失效，整轮停止；必须重新观察，不能改写旧版本号 |
| 未注册 Agent | Rule fallback |
| Controller 结算网络不确定 | 使用同一个 `step_id` 重试；服务端返回首次缓存结果 |
| Action 越界或现金不足 | 统一 decision-policy 调整，并记录 `resolution_adjustments` |

Coordinator 不吞掉 Controller 或 MarketEnv 的结算错误。环境失败时不会伪造下一状态。

## 5. 接入真实模型

实现 `ModelClient` Protocol 即可：

```python
class MyModelClient:
    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        # 将 context 转成提示词，要求模型严格输出 AgentDecision JSON。
        # parsed_output 交给 AgentRuntime 再做 Pydantic 校验。
        return ModelGeneration(
            model_name="provider/model-version",
            parsed_output=parsed_json,
            raw_response=raw_json,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
```

模型输出只能包含经营意图。`episode_id`、公司绑定、回合、状态版本、Action ID 和结算权限均由 Runtime/Gateway/Controller 持有。`raw_response` 应只保存结构化模型响应，不保存隐藏推理过程或凭据。

### 豆包 / 火山方舟

当前实现包含 `DoubaoModelClient`，默认使用：

```text
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=doubao-seed-2-0-lite-260215
ARK_API_KEY=<仅保存在本机环境变量>
```

它使用兼容 OpenAI SDK 的 Responses API，关闭深度思考，不配置 Web Search、MCP 或函数工具。第一次输出如果不是合法 `AgentDecision` JSON，会携带校验错误重新请求一次；仍失败则由 `AgentRuntime` 标记错误，本轮切换为 Controller Rule fallback。

本地完整运行：

```powershell
# PowerShell 1：必须由同一个进程同时承载 8010 和 8011
$env:PYTHONPATH="src"
$env:MARKET_CONTROLLER_TOKEN="local-controller-token"
python -m game_theory_agent.api

# PowerShell 2
$env:PYTHONPATH="src"
$env:MARKET_CONTROLLER_TOKEN="local-controller-token"
$env:ARK_API_KEY="your-local-ark-api-key"
python -m game_theory_agent.run_agents `
  --provider doubao `
  --agent-companies company_A `
  --rounds 5 `
  --seed 42 `
  --market-model balanced
```

先使用 `--provider mock` 可在不消耗模型额度的情况下验证两个 HTTP 服务、Controller Token、Episode 创建、统一结算和 JSONL 写入。需要测试多个豆包 Agent 时使用 `--agent-companies company_A,company_B`；第一阶段建议仍从一个豆包 Agent 开始。

### DeepSeek

DeepSeek 适配器使用 OpenAI 兼容的 Chat Completions API 和官方 JSON Output 模式：

```text
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=<仅保存在本机 .env>
```

启动方式与豆包相同：

```powershell
python -m game_theory_agent.run_agents `
  --provider deepseek `
  --agent-companies company_A `
  --rounds 5 `
  --seed 42
```

提供方和模型的覆盖优先级为：命令行 `--provider/--model` > 环境变量 > 程序默认值。若希望以后默认运行 DeepSeek，只需在 `.env` 中设置：

```dotenv
AGENT_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash
```

两种真实模型都只返回经营计划和 Intent，不能直接执行动作。API 空响应、非法 JSON、Schema 不匹配、超时和网络错误统一进入当前公司的 Rule fallback，并写入 RoundEvent。

## 6. Decision Context 与战略记忆

`decision-context-v1.7.0` 不包含 Episode Seed，也不直接复制完整聊天和全部历史。通信开启时附带当前关闭视图和最近三轮、按公司权限过滤的通信历史；`state_only` 对照会显式清空该历史。每轮发送给模型的是：

```text
meta + identity + 可信 PersonaProfile
+ 当前市场与 Market Regime
+ 单位贡献、保本价格、固定成本、现金跑道和安全投入上限
+ 当前公司和允许范围内的竞争者状态
+ 最近3轮详细计划、请求动作、最终动作和观察结果
+ 最近5轮滚动趋势摘要
+ 最多10个关键事件
+ PlanTracker 阶段、优先级、最低价格、现金储备和最大投入
+ 当前动作约束
```

PersonaProfile 由 Runtime 注入，包含版本化效用权重、时间折扣、风险偏好和预留行为特征。它只影响 Planner 与结果评价，不进入 MarketEnv 公式或 Controller 护栏。当前合作与社会福利能力关闭，详细配置、效用公式和实验方法见 `docs/04-Persona研究/01-persona-research.md`。

Market Regime 的阈值位于 `configs/market_v4.yaml` 的 `agent_context.regime_thresholds`，由代码确定性计算。它只是给 Agent 的高层摘要，不参与需求、成本、消费者选择或状态更新。

`episode-memory-v2.0.0` 分离详细记忆和趋势窗口，避免同时把 `public_history` 与另一份重复历史塞进 Prompt。它记录连续亏损、现金回撤、目标失败与反事实结果，PlanTracker 每轮据此选择 `growth / profit_recovery / liquidity_crisis`。

`result-analysis-v1.3.0` 包含：

- 公司结算前后状态和明确命名的变化量；
- 上一轮市场结果与本轮已结算市场结果；
- 本轮条件和下一决策轮条件，避免轮次语义混用；
- 第一轮没有历史利润时使用 `baseline_unavailable`，不把初始化的 0 当作上一轮利润，也不计入预期 mismatch；
- `observed_directions.capacity` 保留折旧后的物理产能方向，`actual_directions.capacity` 按本轮主动产能投资评价；
- 外部事件与预期匹配情况；
- 预测方向匹配和独立的 `goal_assessment`，预测亏损正确不等于计划成功；
- 同一 State、同一 Seed、其他公司动作固定的现金保护与利润恢复反事实；
- 已运行反事实时 `causal_claim=controlled_same_seed_counterfactual`。

Counterfactual Evaluator 不替代市场结算，也不让 Agent 直接执行动作；它只克隆结算前 State，在相同随机源下比较替代动作并写入回溯。

## 7. RoundEvent 与复现

当前每轮 `agent-round-event-v1.8.0` 日志还包含通信生成与最终决策各自的 `ObservationSnapshot`；启用 Belief 时 Snapshot 同时绑定 Belief State/Hash；旧版字段说明继续兼容。日志包含：

- 冻结前后 State Hash、Joint Action Hash 和随机数摘要；
- 每个 Agent 的观察 Hash、Persona/Profile Hash、模型名、结构化计划与请求动作；
- Persona 各组件得分、逐轮效用、时间折扣和累计折扣效用；
- Intent ID、统一规则调整、最终 CompanyAction；
- 延迟、Token 统计、错误/fallback 和确定性 ResultAnalysis；
- 完整 StepResult 与协调阶段状态。
- 完整 Communication Close、公司级可见视图、通信生成轨迹、最近历史输入和消息回应。

`MarketTransition` 仍是市场 Hash Replay 的权威记录；RoundEvent 是围绕它的 Agent 决策审计记录。Information Replay 从 RoundEvent 的 True State 和 EpisodeManifest 重建每个公司实际视图，并验证 Decision/Communication Context。测试覆盖固定 Seed 下的完整 5 轮闭环，并验证最终 Replay State Hash 与所有公司 Observation Hash 完全一致。

## 8. 当前部署限制

Episode 与 Intent 仍保存在进程内 `SESSIONS`。因此 8010 和 8011 必须由 `python -m game_theory_agent.api` 在同一个 Python 进程中启动，Uvicorn worker 数保持 1。横向扩容前需要先把 Episode 状态、Intent 和幂等结算记录迁移到持久化存储。
