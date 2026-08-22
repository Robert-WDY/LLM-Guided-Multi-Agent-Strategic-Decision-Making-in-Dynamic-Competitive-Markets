# Persona P0/P1/P2 v2 实施说明

本阶段不是把极端异质实验“调成利润更高”，而是保留原始研究基线，增加可分离、可复查的新干预条件。

## 已实现

### P0：版本化 Persona 与通用经济信息

- 市场配置升级为 `market-v4.9.0`，Persona Catalog 升级为 `persona-catalog-v1.1.0`。
- 原 `aggressive`、`conservative`、`balanced`、`selfish_long_term` 未删除或覆盖。
- 新增版本化极端基线：`aggressive_v1_extreme`、`conservative_v1_extreme`、`balanced_v1`、`selfish_long_term_v1`。
- 新增温和实验人格：`disciplined_growth_v1`、`risk_guarded_v1`。
- Decision Support v1.1 新增：当前信息下的需求区间、产能缺口、产能回收期、产能边际价值、事故期望损失区间、当前韧性覆盖、每100万元韧性投入的边际损失减少、增长支出效率历史代理、价格贡献毛利率。
- 每份支持数据声明 `uses_future_rng=false`、`uses_episode_seed=false`。没有固定“推荐韧性目标”。

### P1：语义、诊断与研究指标

- Persona 语义升级为 `economic_v2`：风险厌恶不等于停止投资，增长偏好不等于无条件降价，现金偏好不等于现金越多越好。
- `legacy_v1` 与 `economic_v2` 可切换，支持独立消融。
- 诊断模式只有 `off` 和 `observe`。诊断要求 Agent 回应证据，但 `enforcement=none`，不修改动作、不强制重试。
- Growth Efficiency 没有加入 Persona Utility，只作为 Decision Support 和研究指标。
- 结算后单独记录实际事故损失、未服务订单贡献损失和终局风险调整价值；这些字段不会被重复加入人格效用。
- 四主体 summary 增加研究指标：价格离散与熵、完整动作距离、产能投入标准差/HHI、需求—产能绝对差、最低韧性、条件化未投资轮次、累计未服务需求和 Outside Option。
- 工程硬标准、市场健康指标、无必达方向的研究指标被明确分组。利润和价格多样性不会决定 episode 是否通过。

### P2：位置轮换、干预矩阵与 Holdout 保护

六个干预条件：

| 条件 | Persona | 决策支持 | 新语义 | 诊断 |
|---|---|---|---|---|
| A | 4 balanced | legacy | legacy | off |
| B | 极端异质 | legacy | legacy | off |
| C | 极端异质 | economic | legacy | off |
| D | 极端异质 | economic | economic | off |
| E | 温和异质 | economic | economic | off |
| F | 温和异质 | economic | economic | observe |

异质条件自动生成四种循环位置，每个 Persona 都会占据 company_A–D 各一次。同质条件只运行 rotation 0。

Seed 分区：

- Development：101–110
- Validation：201–220
- Final Holdout：1001–1030

Final Holdout 需要显式 `--confirm-final-holdout`。矩阵清单声明主要实验单位是配对 Seed，决策记录是 episode 内嵌套观察，不是独立样本。自动人格权重搜索保持禁用。

## 使用方式

只生成 Development 实验计划，不调用模型：

```powershell
python -m game_theory_agent.experiments.persona_intervention_matrix `
  --seed-split development `
  --output runs/persona-intervention-development
```

执行温和人格 E/F 条件：

```powershell
python -m game_theory_agent.experiments.persona_intervention_matrix `
  --conditions E_moderate_semantics,F_moderate_diagnostics `
  --seed-split development `
  --provider doubao `
  --execute `
  --output runs/persona-intervention-development
```

执行单个位置条件：

```powershell
python -m game_theory_agent.experiments.four_agent_acceptance `
  --condition F_moderate_diagnostics `
  --rotation-index 2 `
  --seed 201 `
  --provider doubao `
  --rounds 20 `
  --output runs/F-seed201-rotation2
```

## 明确保留的研究边界

- `4 balanced` 的定价趋同仍是合法研究结果，没有加入人为价格扰动。
- 极端人格仍用于 Stress Test 和边界研究。
- 温和人格权重是待验证的起点，不是已被证明的通用参数。
- 诊断器是独立实验条件，不是隐藏规则策略。
- 当前不使用 Development Seed 自动搜索权重；未来搜索必须在 Development 调参、Validation 选择 Pareto 解、Final Holdout 只做一次最终评估。
