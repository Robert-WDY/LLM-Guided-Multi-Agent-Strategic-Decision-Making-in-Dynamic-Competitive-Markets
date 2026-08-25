# Game Theory Agent — Engineering MVP v4

后端市场模型以 `LLM多智能体生鲜配送市场博弈系统_Engineering_MVP技术规格_v4.0.md`
为算法规范，以 `configs/market_v4.yaml` 为唯一参数来源。

当前已经实现：

- 固定点金额、比例和订单状态；
- 2～8 家公司的连续数值联合动作与动态约束；
- 三类消费者、Outside Option 和确定性整数订单分配；
- 产能约束、流动性约束和一次缺货转售；
- Awareness、Service、Reputation、Capacity、Resilience 跨轮更新；
- Risk Signal、重大市场事件、公司事故和销售前维修；
- SplitMix64 组件 Seed、Canonical JSON、State Hash；
- Action/Step 幂等、EpisodeManifest、JSONL 记录和 Hash Replay；
- 终局残值与完整状态不变量。
- 单公司经营模式：玩家控制一家企业，其他企业由确定性规则代理补全联合动作；
- 公司状态诊断、情境化决策建议，以及终局逐轮市场回溯与成败归因。
- 统一决策规则层：人类、规则对手和外部 Agent 的请求都先经过状态护栏与现金约束；
- 独立 Agent Gateway：Agent 只读取观察和提交意图，受保护 Controller 才能执行联合结算；
- Game Theory Lab 研究控制台：实验配置、Human/LLM/Rule Agent 编排、Persona Profile、Agent-scoped Observation、Belief/Advisor、通信可见性、市场结果、Replay 调试和实验报告；
- 市场与研究控制台支持 2–10 家公司；普通无通信、无合作、无战略链的 Episode 可由受保护 Controller 使用确定性规则代理跑完剩余回合，规则覆盖会明确标记且不能冒充真实 LLM 行为；
- 随机 / 固定 Seed 前端入口和 200 Seed 投入档位校准。
- 规则对手使用 Seed 固定的价值型、溢价型、增长型或谨慎型行为，并带可复现的回合扰动；不调用 Agent 或模型；
- 利润计入固定运营和逐单履约成本，声誉采用慢变量更新，完全维修当轮保留残余事故影响；
- Agent 公开观察包含逐轮历史、预警兑现结果、成本与事件传导解释，以及终局企业价值排名。
- Episode 级信息模式：MVP 默认 `perfect` 完全信息，也可切换 `public` 验证未来不完全信息流程；
- Information Architecture Phase A：版本化 Public/Private State、Visibility Policy、公司级 Observation Hash、Intent 视图绑定与独立 Information Replay；`public` 已实现零对手财务/运营泄漏；
- Belief MVP Phase B：可选 `public_action_v1` 仅从已结算公开价格生成对手下一轮降价/持平/涨价概率，带 company-scoped Ledger、Belief Hash、Belief Replay 与 Accuracy/Brier/Log Loss；默认 `off` 保留无信念基线；
- 不完全信息 P0–P5：严格 `PublicState / PrivateState / ObservationEnvelope`，可选 `public_action_signal_v2` 将实际可见的结构化非绑定声明按历史可靠度加入信念；`bayesian_price_v1` 提供带 Advice Hash/Replay 的非绑定 Approximate Bayesian Price Response，不读取对手隐藏状态；
- Game Theory Enhancement：`public_strategy_v1` 从公开历史形成 growth/profit/defensive/cooperative 对手模型，`strategy_utility_v1` 推断六项效用权重，`bayesian_strategy_v2` 在有限价格动作上计算 Approximate Bayesian Best Response；重复博弈层从合作记忆生成 Tit-for-Tat/Grim/Generous 建议，全部带 Hash/Replay 且不直接修改市场；
- Stage 6 Strategic Decision Reliability P0：使用真实 `MarketEnv` 对 7–10 个候选执行多轮、多情景反事实，按企业价值、累计利润、风险损失和 Persona Utility 排序；权威 Rollout 仅作为离线 Oracle，不进入公开信息 Agent 上下文，并提供 Known/Mixed/100 Unknown/30 Holdout 对手基准和默认拒绝真实模型调用的成本门禁；
- Stage 6.1 Public Rollout v1：`public_rollout_v3` 仅从公共状态、本公司私有状态、Belief 和公开 Opponent Model 重建预测市场，以“Rule 运营基线 + 单一战略增量”执行在线长期规划；支持 Observation/Advice Hash 与 Advisor Replay，5 个共同 Seed 的 10 轮 Rule 对照全部通过企业价值、最坏 Seed 和权威 Regret 晋升门禁；
- Stage 6.2 Strategic Ablation：以共同市场 Seed 和共同 Advisor 随机场景完成 Rule、Persona Planner、Belief、Opponent v1、Opponent v2 五组 × 三人格 × Known/Mixed/Holdout 的 135 局零模型消融；Persona Planner 在 27/27 配对中提高企业价值但激进人格 Regret 未过门禁，Belief/v1 存在长期价值或尾部退化，v2 为 0 组改善、3 组恶化，全部保持非默认研究 treatment；
- Stage 6.3 Objective Calibration：Rollout 新增最强对手价值、竞争差距和名次标签，以 243 局零模型实验比较绝对价值、Persona 和四档竞争权重；开发集为三人格选择的固定权重在保留集全部失败，Belief 路径中 33.33% 出现“一步更优、长期更差”，利润人格 v1 门控也未过尾部与竞争门禁，所有候选保持离线；
- Stage 6.4 Pareto Reliability：新增价值/竞争/尾部/人格四维约束和安全 Pareto 前沿，修复 Rule 与处理组信息权限不一致及 Regret 候选口径错误；10 个共同 Seed、720 局的开发集/保留集验证中，三人格的企业价值、竞争差距、第一名率和两轮 Regret 全部门禁通过，新增可回放的公开信息 `pareto_rollout_v4`，真实模型调用与费用仍为 0；
- Stage 6.5 Real Advisor Adoption：新增带 Hash 的 Advisor Adoption Trace 和只在预注册轮调用 Provider 的成本门禁；65 次豆包真实调用表明 Pareto 使 13/13 配对改变动作、推荐采纳率 84.62%，但短期逐利与风险防御存在负尾部，阶段研究门禁未通过并按规则停止在自适应 3/5 Seeds，没有继续消耗完整 10-Seed 预算；
- Stage 6.6 Failure Forensics：零 Token 重放 Stage 6.5 的 65 个 Episode，并对五个负配对执行固定后续动作带反事实；五例均精确采纳且无 Controller 经济调整，主要归因为四例 Forecast/内生反应误差和一例候选集合覆盖不足，因此暂缓 Advice Contract v2 和 30 次真实复测；
- Stage 6.6 Reliable Repair：保留 v3/v4 语义和回放，新增公开信息 `pareto_reliable_v5`；候选集合加入零可选投入、利润恢复和风险缓冲，低对手置信度或候选差距小于预测不确定度时自动放弃 Planner 动作并退回情境安全动作。五个已知失败状态的冻结动作带 Auto-execute 全部不低于无建议基线，三个正收益、两个持平，新增真实模型调用和 Token 均为 0；
- 固定 Observe → Plan → Intent 工作流、结构化 AgentDecision、短期 Episode Memory 与确定性 ResultAnalyzer；
- Agent Context v1.2：隐藏随机 Seed，增加单位经济、现金跑道、PlanTracker、最近3轮详细记录、5轮趋势摘要和关键事件；
- ResultAnalysis v1.3 分离预测准确率与目标达成情况，并记录同状态、同随机源的现金保护和利润恢复反事实；
- Persona Catalog v1.0：人格以效用权重、时间折扣和风险/行为特征进入 Agent Planner，不改变 MarketEnv 或统一安全护栏；
- 支持无人格、激进、保守、均衡、长期自利和短期逐利 Profile，并记录确定性逐轮人格效用、折扣累计效用与 Profile Hash；
- Interaction MVP 提供非约束公开消息、一对一私信、公司级可见性、Communication Close 与独立 Interaction Replay；
- Cooperation MVP v1 只启用 `Shared Resilience Contribution`：私密提议、接受/拒绝、非约束承诺、真实贡献、公共韧性、履约/背离、可信度与独立 Cooperation Replay；联合定价、转账、联盟、物流和产能共享仍禁用；详见 `docs/cooperation-mvp-v1.md`；
- Controller Policy v1.1 预留固定运营成本、保护最低单位贡献，并在恢复阶段阻止继续降价；
- RoundCoordinator 并发冻结观察、逐公司超时/fallback、统一结算，并输出可审计 RoundEvent JSONL；
- 内置 MockModelClient 可在未接入真实模型时验证 `1 Agent + 3 Rule` 与多 Agent 联合动作闭环。
- 豆包/火山方舟 `DoubaoModelClient` 与 DeepSeek `DeepSeekModelClient`：统一结构化决策契约、一次格式修复和规则 fallback。
- 可选择均衡、价格敏感、品质偏好、服务偏好或 Seed 随机市场；市场公开需求偏差、价格锚点与合理价格区间。
- 品质市场显著强化品牌知名度与声誉，服务市场显著强化当期服务与历史缺货惩罚，不再仅靠消费者分群比例形成细微差别。
- Episode 支持 `5 / 10 / 15 / 20` 轮、随机或固定 uint64 Seed，并提供 Agent 只读选项发现接口；创建与结算仍由私有 Controller 负责。
- 对手报价围绕市场锚点、上一轮成交价和成本底线重新计算，不再把自身上一轮报价当作只能上涨的硬下限。
- 终局同时提供综合价值榜和总资产榜、四家公司逐轮综合价值折线，以及与冠军逐项对比的结果解释。
- `settled_market` 与 `public_history[].market` 使用同轮事件、情绪和供应成本口径；顶层 `state.market` 表示下一决策轮当前条件。

前端已升级为 LLM 多智能体博弈实验控制台，不保留独立市场公式。高级真实 Episode 必须使用受保护 Controller 契约；无后端时可加载明确标识、不会冒充真实模型结果的 Research Demo。

## 项目结构

```text
configs/                         市场唯一参数源
src/game_theory_agent/           Python 市场引擎、决策规则与 API
tests/                           确定性、动力学、回放、接口和校准测试
frontend/                        React/Vinext 可视化控制台
docs/                            Agent 接口、市场模型与回溯协议
examples/                        最小运行示例
```

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

分别启动 API 和前端：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.api

cd frontend
npm ci
npm run lint
npm run dev
```

默认端口为：前端 `3210`、私有市场引擎 `8010`、Agent Gateway `8011`。启动 Controller 前设置高熵令牌：

```powershell
$env:MARKET_CONTROLLER_TOKEN="请替换为随机高熵令牌"
```

Agent 接口、权限边界和请求示例见 [docs/agent-gateway-api.md](docs/agent-gateway-api.md)。在线 OpenAPI 位于 `http://127.0.0.1:8011/docs`。
进入不完全信息前的 Observation Split、Visibility Policy、View Hash 和 Information Replay 见 [docs/information-architecture-refactor.md](docs/information-architecture-refactor.md)。
确定性公开动作信念、Belief Hash、Belief Replay 与验收见 [docs/belief-mvp.md](docs/belief-mvp.md)。
真实豆包模型的固定状态重复和 3 个共同 Seed Belief OFF/ON 配对 Pilot 见 [docs/belief-real-paired-pilot.md](docs/belief-real-paired-pilot.md)。
P0–P5 的严格信息契约、通信信号信念、Bayesian Advisor 与五层 Replay 验收见 [docs/incomplete-information-p0-p5.md](docs/incomplete-information-p0-p5.md)。

不完全信息市场中的单一完全信息 Agent、跨人格配对夺冠实验见 [docs/privileged-information-persona-experiment.md](docs/privileged-information-persona-experiment.md)。
Opponent Modeling、Utility Inference、Advisor v2、Repeated Game Strategy 与完整 GameTheory Replay/基准见 [docs/game-theory-enhancement.md](docs/game-theory-enhancement.md)。
真实 LLM 的 Persona/Belief/Opponent Model/Utility+Advisor 四组消融、真实市场反事实、Token 与单 Seed Pilot 结果见 [docs/stage51-real-game-theory-evaluation.md](docs/stage51-real-game-theory-evaluation.md)。
Stage 6 的长期真实市场 Rollout、Persona Planning、Opponent Holdout 和零 Token 成本门禁见 [docs/stage6-strategic-decision-reliability.md](docs/stage6-strategic-decision-reliability.md)。
Stage 6.1 的公开信息预测市场、在线候选语义、失败修复、五 Seed Rule 对照与零成本验收见 [docs/stage6-public-rollout-v1.md](docs/stage6-public-rollout-v1.md)。
Stage 6.2 的共同随机数修复、五层下游消融、人格/对手池拆分、晋升结论与零成本验收见 [docs/stage6.2-strategic-ablation.md](docs/stage6.2-strategic-ablation.md)。
Stage 6.3 的绝对价值/竞争夺冠目标拆分、开发集与保留集校准、Belief 路径归因和零成本验收见 [docs/stage6.3-objective-calibration.md](docs/stage6.3-objective-calibration.md)。
Stage 6.4 的约束式 Pareto 选择、公平对照修复、长期 Regret、10-Seed 保留集结果和 `pareto_rollout_v4` 在线晋升见 [docs/stage6.4-pareto-reliability.md](docs/stage6.4-pareto-reliability.md)。
Stage 6.5 的真实模型采纳 Trace、低 Token 渐进实验、Belief/旧 Advisor/Pareto 对照、负尾部和停止规则见 [docs/stage6.5-real-advisor-adoption.md](docs/stage6.5-real-advisor-adoption.md)。
Stage 6.6 的五个失败状态、Planner/Adoption/Execution/Forecast 归因、65 条历史映射、全层 Replay 与修复优先级见 [docs/stage6.6-failure-forensics.md](docs/stage6.6-failure-forensics.md)。
多 Agent 研究控制台的信息架构、真实/演示边界、八个工作区与前端验收见 [docs/frontend-research-dashboard.md](docs/frontend-research-dashboard.md)。
单 Agent 与多 Agent 的运行时、协调器、日志和接入示例见 [docs/agent-runtime-and-orchestration.md](docs/agent-runtime-and-orchestration.md)。
人格配置、效用公式、实验隔离和非合作阶段边界见 [docs/persona-research.md](docs/persona-research.md)。
阶段 2 通信状态机、可见性和验收设计见 [docs/phase2-interaction-mvp-design.md](docs/phase2-interaction-mvp-design.md)，P1/P2/P3 实现结果见 [docs/phase2-p1-p2-p3-implementation.md](docs/phase2-p1-p2-p3-implementation.md)。
真实豆包模型的单 Seed 基线见 [docs/phase2-real-llm-smoke-seed810.md](docs/phase2-real-llm-smoke-seed810.md)，5 个共同 Seed、三种通信条件及固定状态反事实验收见 [docs/phase2-real-llm-5seed-smoke.md](docs/phase2-real-llm-5seed-smoke.md)。
市场模型、榜单口径与回溯字段见 [docs/market-models-and-ranking.md](docs/market-models-and-ranking.md)。
四人协作的分支、Review、模块边界和合并检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。每次 Pull Request 会自动运行后端测试、前端 lint 和生产构建。

## 模型 Agent 冒烟运行

安装后编辑本地 `.env`，填入 `ARK_API_KEY` 和/或 `DEEPSEEK_API_KEY`。该文件已被 Git 忽略。`AGENT_PROVIDER` 可设置为 `doubao`、`deepseek` 或 `mock`。

在第一个 PowerShell 启动同进程的市场引擎和 Agent Gateway：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.api
```

第二个 PowerShell 运行 Agent；程序会自动加载同一个 `.env`：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.run_agents --provider doubao --rounds 5 --seed 42

# 或切换为 DeepSeek
python -m game_theory_agent.run_agents --provider deepseek --rounds 5 --seed 42

# 可复现随机对手的20回合实验
python -m game_theory_agent.run_agents --provider doubao --rounds 20 --seed 42 --opponent-policy uniform-random --opponent-seed 9001

# 同一市场 Seed 下运行长期自利人格
python -m game_theory_agent.run_agents --provider deepseek --persona selfish_long_term --rounds 20 --seed 42

# 多 Agent 使用不同人格；未覆盖的模型公司使用 --persona 的默认值
python -m game_theory_agent.run_agents --provider mock --agent-companies company_A,company_B --persona balanced --persona-map company_A=profit_myopic,company_B=conservative --rounds 5 --seed 42
```

也可以省略 `--provider`，直接修改 `.env` 中的 `AGENT_PROVIDER`。`--model` 可临时覆盖对应的 `ARK_MODEL` 或 `DEEPSEEK_MODEL`。默认只有 `company_A` 使用所选模型，另外三家公司使用规则 fallback。运行日志写入 `runs/<episode_id>/agent-rounds.jsonl`，真实密钥不得写入仓库。

初始状态的 `round=1` 表示等待第一轮决策。每次调用：

```python
result = env.step(
    step_id=f"{state.episode_id}:{state.round}:{state.state_version}",
    joint_action=actions,
)
state = result.state_after
```

相同配置、初始状态、联合动作、Seed 和环境版本会得到相同的 State Hash；
相同动作在不同轮次会受到历史状态和独立轮次随机组件影响，不再机械地产生相同结果。
