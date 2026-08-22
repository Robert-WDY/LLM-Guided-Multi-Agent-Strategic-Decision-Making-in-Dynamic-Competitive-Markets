# Persona Research MVP：非合作阶段

当前阶段将 Persona 实现为 Agent 侧、版本化、可审计的实验输入：

```text
PersonaProfile
→ DecisionContext / Prompt
→ AgentDecision
→ 通用 Controller 护栏
→ MarketEnv.step
→ PersonaUtilityAssessment
→ EpisodeMemory / RoundEvent
```

`MarketEnv` 不读取 PersonaProfile，不为任何人格修改需求、成本、事件、产能、残值或安全边界。标准 `run_agents` 路径也不把人格写入 `CompanyState`，因此同一市场配置、Seed 和公司集合的初始 State Hash 不随 Agent 人格变化。

旧 `POST /api/episodes.personas` 字段暂时保留，用于历史状态兼容；正式人格配对实验应使用 Runtime PersonaProfile 和 `agent_configs`。

## 1. 配置

人格目录位于 `configs/market_v4.yaml` 的 `persona_utilities`，目录版本为 `persona-catalog-v1.0.0`。所有 Profile 包含：

- 结果权重：`profit / share / growth / stability / cash / reputation / resilience`；
- 预留结果：`social_welfare / cooperation_reputation`；
- 特征：`time_discount / risk_aversion / reciprocity / commitment_honesty / opportunism`；
- 中文标签、可执行目标、版本和 Profile Hash。

当前可选 Profile：

| ID | 含义 |
| --- | --- |
| `none` | 历史无人格基线；当前配置为纯当轮利润权重 |
| `aggressive` | 偏市场份额和增长，风险厌恶较低 |
| `conservative` | 偏现金、稳定性和韧性 |
| `balanced` | 平衡利润、份额、现金、声誉与韧性 |
| `selfish_long_term` | 只关心本公司长期价值，重视现金和长期资产 |
| `profit_myopic` | 90% 当期利润、10% 现金、低时间折扣，不奖励份额增长或长期资产 |

所有结果权重使用整数 ppm 且总和必须为 `1000000`。配置加载器会拒绝缺字段、越界特征、权重和错误以及未启用能力的非零权重。

## 2. Persona 与 Cooperation MVP v1 的边界

默认 Episode 仍是：

```text
social_welfare = false
cooperation = false
```

因此所有 Profile 的 `social_welfare` 和 `cooperation_reputation` 权重仍必须为零。

显式启用 `cooperation_mode=shared_resilience_v1` 后，只开放私密韧性提议、非约束承诺和真实公共韧性贡献。它不启用共同效用、合作人格、联合定价或社会福利公式；Persona 仍只影响 Agent 如何权衡当期私人成本、未来风险和可信度信息，不修改市场公式或执行护栏。详见 `docs/05-交互合作与综合实验/05-cooperation-mvp-v1.md`。

`reciprocity / commitment_honesty / opportunism` 已进入版本化 Profile，但 Cooperation MVP v1 尚未给它们增加 Controller 侧效用或强制执行语义，也不直接参与市场结算。

## 3. 效用计算

逐轮组件全部使用整数 ppm：

```text
ProfitScore    = clip(RoundProfit / ProfitScale, -1, 1)
ShareScore     = MarketShare
GrowthScore    = clip((Share_t - Share_t-1) / 0.10, -1, 1)
StabilityScore = 1 - min(PopulationStdDev(last_3_profits) / ProfitScale, 1)
CashScore      = clip(Cash / (2 × InitialCash), 0, 1)
Reputation     = state reputation ppm
Resilience     = state resilience ppm
```

少于两个利润样本时 `StabilityScore = 1000000`。标准差、加权和与时间折扣均使用整数运算。

```text
RoundUtility = Σ(weight_k × component_k) / 1000000
DiscountedRoundUtility_t = gamma^t × RoundUtility_t
```

终局日志同时记录权威 `terminal_enterprise_value_cents`，但本版本没有把它重复加入逐轮加权分数，避免现金、声誉和韧性被二次计权。

效用评价基于 Controller 最终执行动作的实际结果；如果模型失败而切换 Rule fallback，日志仍会评价 fallback 结果是否符合原定人格。

## 4. Prompt 与硬约束

`decision-context-v1.4.0` 包含完整可信 `persona_profile` 和 `full / state_only` 上下文实验模式。模型不能自己选择或修改 Persona。

`current_plan` 只表示财务状态和通用硬约束，不再把“增长”作为所有人格共同目标。价格安全线、现金储备、末轮长期投资禁用、资产饱和限制和事故维修规范仍由统一 Controller 执行，任何人格都不能绕过。

## 5. 运行

单一 Persona：

```powershell
python -m game_theory_agent.run_agents `
  --provider deepseek `
  --agent-companies company_A `
  --persona selfish_long_term `
  --rounds 20 `
  --seed 42
```

不同公司使用不同 Persona：

```powershell
python -m game_theory_agent.run_agents `
  --provider mock `
  --agent-companies company_A,company_B `
  --persona balanced `
  --persona-map company_A=profit_myopic,company_B=conservative `
  --rounds 5 `
  --seed 42
```

`--persona` 是所有模型公司的默认值；`--persona-map` 只覆盖 `--agent-companies` 中的公司。Uniform-random 对手使用 `none` Profile，未注册 Runtime 的 Controller Rule 对手没有 Agent Persona。

## 6. 审计与复现

EpisodeManifest 的 `agent_configs` 记录：

- Agent ID、Provider 和模型名；
- Persona 完整配置、目录版本和 Profile Hash；
- 决策超时。

`agent-round-event-v1.2.0` 每轮记录：

- Persona ID、目录版本和 Profile Hash；
- 原始请求与 Controller 最终动作；
- 各效用组件和加权贡献；
- 当前折扣倍率、逐轮效用和累计折扣效用；
- 当前能力是否支持社会福利或合作。

## 7. 配对实验

人格实验应固定：

- 市场配置和市场 Seed；
- 模型提供方、模型版本、Prompt 版本和信息模式；
- 对手策略、对手 Seed 和公司位置；
- 决策超时与模型采样设置。

只替换焦点 Runtime 的 PersonaProfile。第一轮观察的 State Hash 必须完全相同；闭环运行从焦点 Agent 产生不同动作后，后续状态允许自然分化。

本阶段推荐比较动作分布、终局企业价值、利润波动、份额增长、风险预警后的韧性投入、维修策略和累计人格效用。合作率、搭便车收益、背叛率和社会福利留到合作机制真正进入动作与状态之后。

## 8. P0：逐样本反事实与遗憾值

`persona_pilot_counterfactual` 会结算每一个重复动作，不再只用一个代表动作推断整组表现。核心指标为：

```text
Regret = 所有候选动作中的最高人格效用 - 当前动作的人格效用
StrictOptimalRate = Regret 为零的自身动作数 / 自身动作总数
DeltaVsNone = 人格动作平均效用 - 无人格动作平均效用
```

`alignment_vs_other_mean_ppm` 继续保留，但它只表示高于其他动作平均值，不能解释为严格最优。

```powershell
python -m game_theory_agent.experiments.persona_pilot_counterfactual `
  runs/persona-pilot-doubao-example
```

结果写入 `counterfactual_v2.json`，旧版 `counterfactual.json` 不被覆盖。

## 9. P1：多状态、基线与消融

冻结上下文实验支持多个独立市场 Seed、所有场景的 `none` 基线以及 Persona Contract 消融：

- `full`：完整人格；
- `label_only`：只保留人格标签；
- `objective_only`：只保留目标文本；
- `weights_only`：只保留效用权重；
- `traits_only`：只保留风险、时间折扣等 traits。

```powershell
python -m game_theory_agent.experiments.persona_pilot `
  --provider doubao `
  --personas none,aggressive,conservative,selfish_long_term,profit_myopic `
  --ablations full,objective_only,weights_only,traits_only,label_only `
  --scenarios normal,risk_warning,financial_stress `
  --market-seeds 41,42,43,44,45 `
  --repetitions 3 `
  --temperature 0 `
  --top-p 1
```

`manifest.json` 固化所有初始状态、采样参数和实验条件。Provider 当前没有通用可移植的生成 Seed，因此记录为 `provider_seed: null`。

## 10. P2：多回合人格轨迹

多回合实验让焦点 Agent 在同一 Episode 中保留记忆和折扣效用，其他公司使用同一确定性规则策略。由于规则对手根据当前状态响应，不同人格造成状态分叉后，对手的后续实际动作可以自然不同。

```powershell
python -m game_theory_agent.experiments.persona_multiround `
  --provider doubao `
  --personas none,balanced,aggressive,conservative,selfish_long_term,profit_myopic `
  --market-seeds 41,42,43 `
  --rounds 5 `
  --temperature 0 `
  --top-p 1
```

输出包括逐轮 `rounds.jsonl` 和汇总 `summary.json`。主要比较累计利润、最终与最低现金、最终份额、声誉、韧性、总主动投入、累计折扣人格效用和终局企业价值。
