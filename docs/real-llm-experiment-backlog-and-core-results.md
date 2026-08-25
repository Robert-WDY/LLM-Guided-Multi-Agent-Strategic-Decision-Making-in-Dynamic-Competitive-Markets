# 真实 LLM 实验缺口与核心验证结果

日期：2026-08-25。

## 1. 为什么不把所有旧计划重新跑一遍

项目已经积累了大量真实模型证据：多 Agent 构成、人格异质性、通信 Smoke、30 个合作 Episode、Belief 配对、Stage 5.1 博弈模块消融、Stage 6.5 的 65 次 Advisor Adoption、完全信息先导实验和不可变版本真实 Smoke。继续重复这些矩阵只会增加费用，不能回答新的工程决策。

本轮先将“没有做过”和“做过但证据仍不足”分开，再选择会直接影响下一步算法修改的两个问题：

1. `pareto_reliable_v5` 修复后，真实 LLM 是否会采纳，并消除五个历史失败状态的负尾部；
2. 高低信誉是否会在相同合作提议下改变真实 LLM 的接受和贡献，以及这种效果是否受人格影响。

## 2. 尚待实验问题总表

| 研究问题 | 已有真实证据 | 仍缺少什么 | 最小正式设计 | 预计调用 | 优先级与状态 |
|---|---|---|---|---:|---|
| v5 Advisor 真实采纳与尾部收益 | v4 已有 65 次调用；v5 只有冻结输出零 Token 修复 | 同一冻结状态下无建议与 v5 的新模型配对 | 5 个失败状态 × 2 条件 | 10 | P0，本轮已执行 |
| Credibility 的独立因果作用 | 30 个合作 Episode 证明完整历史会影响行为 | 固定提议下只改变信誉，并跨人格重复 | 2 人格 × 3 条件 × 2 次 | 12 | P0，本轮已执行 |
| Belief 更新器泛化 | 3 个共同 Seed、固定状态和 Stage 6.5 消融已有方向证据 | 累计频率、滑动窗口、遗忘因子和 Placebo 的 10 Seed 对照 | 10 Seed × 3 更新器，单付费轮 | 30 | P1；先等 v5 尾部修复 |
| Opponent Model 的 Known/Mixed/Holdout 泛化 | 五 Seed 中未证明优于动作预测，真实 LLM 很少明确引用 Type | 未知混合效用、校准分数和下游收益 | Known/Mixed/Holdout × 3 Seed × ON/OFF | 18 | P1；先离线提高预测器 |
| 通信信号进入 Bayesian Belief | 旧固定消息实验证明消息能改变动作 | 声明—事实不一致、欺骗和信誉似然的多次配对 | 4 信号 × 3 重复 × 2 对手类型 | 24 | P1；需先冻结 Signal Likelihood |
| 合作人格长期重复博弈 | Balanced 条件下已有 30 个 Episode；本轮覆盖均衡与短期逐利 | 专用 Cooperator、Conditional Cooperator、Free Rider、Retaliator 的数学效用尚未进入 Catalog | 4 人格 × 3 Seed × 2 条件，单付费轮或稀疏付费 | 24 | P1；当前不具备实验前提 |
| 信誉→接受→贡献→福利的多轮链 | 当前只有完整历史条件对照和本轮固定状态因果 | 同一 Agent 的连续信誉更新和真实后续响应 | 2 LLM + 2 Rule、5 Seed、10 轮、隔轮付费 | 50 | P1；固定状态门禁通过后再跑 |
| 完全信息 Agent 的多 Seed收益 | 3 种人格的单 Seed 先导已否定“所有人格都必然第一” | 效应大小、位置轮换和五人格覆盖 | 5 人格 × 3 Seed × 2 信息条件 | 30 Episode/300 决策 | P2；强命题已被否定，不值得当前扩跑 |
| 跨模型稳健性 | 历史 Doubao 与本轮 DeepSeek 分别有结果 | 同一冻结输入在 Doubao、DeepSeek、GPT 上的差异 | 5 状态 × 3 模型 | 15 | P2；缺少全部 Provider 凭据与固定快照 |
| 未知 Seed 的 v5 长期泛化 | v4 已知失败、v5 本轮仍有一个负尾部 | 修复后在未知 Seed、Mixed/Holdout 上验证 | 3 Persona × 3 对手类 × 5 Seed，稀疏付费 | 至少 45 | 暂停；严格尾部门禁尚未通过 |
| P3–P5 Population、Matchmaker、Mutation/Promotion | 尚未实现 | 先有工程合同与零 Token验收 | 实现后再设计 | 0（当前） | 不是当前可执行实验 |
| 现实数据校准 | 当前结论均来自合成市场 | 价格弹性、成本、事故概率的外部数据集 | 数据校准后做合成/现实参数对照 | 待定 | P3；当前机制仍在收口 |

## 3. 本轮预注册核心实验

### 3.1 实验 A：v5 五个失败状态真实配对

每个配对固定：

- Stage 6.5 第 7 轮前的 Market State；
- Seed、Persona、Belief、Opponent Model、Utility Inference；
- 第 7–10 轮全部对手动作及第 8–10 轮焦点 Agent 动作；
- 模型 `deepseek-v4-flash`、temperature=0、top_p=1；
- Schema 重试 0，Transport 重试 0。

只改变：

- Control：Belief + Opponent Model + Utility，无 Advisor；
- Treatment：相同输入 + `pareto_reliable_v5`。

每个新 LLM 动作经过同一个 Controller Resolver，再放入同一冻结动作带结算。主要指标是 Treatment−Control 的三轮终局企业价值；同时记录动作变化、Advice Adoption、Target Alignment 和 Provider Audit。

严格门禁预先定义为：5/5 配对完整、平均企业价值不下降、最差配对不下降，并且建议至少改变一次动作或被明确采纳。

### 3.2 实验 B：信誉 × 人格合作因果

固定同一个市场状态、同一个贡献请求和相同可见字段，只改变：

- 无消息；
- 高信誉公司提出贡献 100 万分；
- 低信誉公司提出相同贡献 100 万分。

人格选择当前 Catalog 已真实实现且差异最大的两种：

- `balanced_v1`：互惠 300,000 ppm、承诺诚实 650,000 ppm；
- `profit_myopic`：互惠 0、承诺诚实 100,000 ppm、机会主义 950,000 ppm。

每个单元重复两次并轮换调用顺序，共 12 次。指标为消息接受率、拒绝率、实际贡献和高信誉相对低信誉的差值。两次重复只能作为方向性证据，不能做显著性推断。

### 3.3 费用和可复现约束

总计划严格为 22 次调用。每次预留最多 32K 输入、4K 输出，保守费用上限 0.08 元；总预算上限 1.76 元。价格快照采用 [DeepSeek 官方计费说明](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) 中 `deepseek-v4-flash` 的输入 1 元/百万 Token、输出 2 元/百万 Token，并按高峰双倍费率保守计量。

每次调用都绑定不可变 Agent Version，保存 Version ID、Manifest Hash、Input Snapshot Hash、原始输出、解析动作、Controller Final Action、Token、Request ID、响应模型和 system fingerprint。远程模型属于 `C_recorded_output`：可重放输出后的处理，不能保证再次请求生成相同文本。

## 4. 实际结果

### 4.1 工程结果

- 22/22 调用成功；
- 22/22 Provider Audit 完整；
- 22/22 不可变版本归属完整；
- 实际使用 215,988 输入 Token、10,721 输出 Token；
- Schema 重试 0，Transport 重试 0；
- 高峰双倍费率保守估算 0.474860 元，低于 1.76 元上限；
- 共调用 6 个不同 Agent Version，均只晋升到 `candidate`。

### 4.2 v5 战略结果

真实模型在 5/5 Treatment 中都精确执行 v5 推荐，5/5 相对 Control 改变动作。企业价值差为：

| Seed | Persona | Treatment−Control 企业价值（分） | Adoption |
|---:|---|---:|---|
| 1 | risk_guarded_v1 | +106,305 | exact_action |
| 2 | profit_myopic | 0 | exact_action |
| 3 | risk_guarded_v1 | +778,243 | exact_action |
| 4 | profit_myopic | +1,318,811 | exact_action |
| 4 | risk_guarded_v1 | −85,845 | exact_action |

平均改善 423,503 分，3 正、1 平、1 负。建议采纳和平均收益门禁通过，但最差价值不下降门禁失败，因此战略研究总门禁不通过。

唯一负配对中，v5 选择零增量 `status_quo`，Control 保留 500,000 分服务投入；冻结动作带下 Control 企业价值高 85,845 分。这说明 v5 已解决大部分过度投资和利润恢复问题，但“安全退守等于所有可选投入归零”仍可能删除小额有效服务投入。下一步应先做零 Token 候选基线修复，而不是扩大未知 Seed 真实实验。

### 4.3 信誉与人格结果

| Persona | 条件 | 接受率 | 平均贡献（分） |
|---|---|---:|---:|
| balanced_v1 | 无消息 | 0% | 250,000 |
| balanced_v1 | 高信誉提议 | 50% | 500,000 |
| balanced_v1 | 低信誉提议 | 0% | 0 |
| profit_myopic | 无消息 | 0% | 0 |
| profit_myopic | 高信誉提议 | 0% | 0 |
| profit_myopic | 低信誉提议 | 0% | 0 |

均衡人格中，高信誉相对低信誉提高接受率 50 个百分点、平均贡献 500,000 分；相对无消息提高贡献 250,000 分。短期逐利人格在三种条件下都不贡献，并明确说明公共贡献没有当期利润回报。

这支持两个方向性结论：信誉确实能进入合作决策，但作用不是无条件的；人格会调节信誉效果，高信誉不能克服极端短期逐利偏好。由于每格只有两次，不能声称稳定概率或统计显著。

## 5. 本轮暴露的问题

首次 22 次调用全部结束后，汇总器在生成 Report Hash 时失败。原因是项目 Canonical JSON 禁止浮点，而汇总费用使用了浮点元。所有逐条结果、Raw Output 和 Token 已在每次调用后落盘，因此没有重新调用模型。修复为同时保存微元整数和六位定点字符串，再使用 `--resume` 从 22 条现有记录重建汇总。

修复汇总器会改变当前源码 Hash，但六个被调用 Agent Version 仍指向调用当时的原始 Source Bundle；`registered-versions.json` 在恢复时不覆盖。这保持了不可变版本的真实语义，也证明中途汇总错误不需要消耗第二遍 Provider Token。

## 6. 下一步实验门槛

当前不应直接跑 10 Seed 或 45 次未知对手扩展。建议顺序是：

1. 零 Token 修复 v5 退守候选，使其保留有正边际价值的小额服务/运营投入；
2. 在唯一负状态和至少两个已通过状态上做冻结动作带 Auto-execute，要求最差不低于 Control；
3. 只复测受修复影响的 3 个状态，每状态 Control/Treatment 各一次，最多 6 次真实调用；
4. 为 Cooperator、Conditional Cooperator、Free Rider、Retaliator 建立数学效用和版本化 Persona 后，再运行长期信誉合作矩阵；
5. 上述门禁通过后，才扩展 Belief 更新器和 Known/Mixed/Holdout 泛化。

## 7. v6 零 Token 修复更新

上述第 1–2 步已完成，且没有新增真实模型调用。新版本 `pareto_reliable_v6` 把 `status_quo` 改为逐项及组合边际收益筛选后的经营基线；同五个真实 LLM Control 的冻结动作带结果为 5/5 严格提高，平均 +453,268 分，最差 +106,305 分。原风险防御 Seed 4 负配对从 −85,845 分修复为 +306,173 分。

但 20 个单项公开预测与真实冻结动作带的方向一致率只有 70%，所以不应立刻把“最多 6 次真实调用”扩大成未知 Seed 正式实验。下一步先零 Token 校准事件条件下的韧性和服务预测；只有内部预测门禁提高后，才对受影响状态做不超过 6 次的真实 v6 采纳复测。完整结果见 `docs/stage6.7-marginal-candidate-repair.md` 和 `runs/stage6.7-marginal-candidate-repair/summary.json`。

正式产物：

- `runs/core-real-llm-validation-20260825/summary.json`；
- `runs/core-real-llm-validation-20260825/rows.json`；
- `runs/core-real-llm-validation-20260825/registered-versions.json`。
