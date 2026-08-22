# Stage 5.1：真实 LLM Game Theory Evaluation

日期：2026-08-21。

## 1. 研究问题与冻结边界

本阶段不继续增加 Opponent Model、Utility Inference 或 Advisor，而是在冻结 `public_strategy_v1 / strategy_utility_v1 / bayesian_strategy_v2` 的前提下增加 Evaluation Layer。当前先回答三个问题：

1. 在 Persona、模型、市场、Seed 和 Rule 对手相同时，Action Belief 与 Opponent Model 是否改善真实 Agent 的市场结果；
2. Opponent Model 相对只预测动作是否产生额外影响；
3. Advisor 是否改善真实 MarketEnv 的 Profit、Enterprise Value、Share、Risk 与一步反事实 Regret，而不再用 Advisor 自己定义的 Utility Proxy 证明自己有效。

不求 Nash，不做 MCTS，不增加第二个 Advisor。统计单位固定为共同 Seed，不能把单轮决策当成独立样本。

## 2. 四组消融与隔离

每个 Seed 运行 1 个真实 LLM 公司 A 和 3 个确定性 Rule 对手，模型、Persona、Market、Seed、采样参数和对手均相同，只改变传给 A 的战略信息：

| 组 | Belief | Opponent Model | Utility | Advisor |
|---|---|---|---|---|
| A Persona Only | off | off | off | off |
| B + Action Belief | `public_action_v1` | off | off | off |
| C + Opponent Model | `public_action_v1` | `public_strategy_v1` | off | off |
| D + Utility + Advisor | `public_action_v1` | `public_strategy_v1` | `strategy_utility_v1` | `bayesian_strategy_v2` |

条件调用顺序按 Seed 循环移位，降低供应商时间漂移与顺序效应。每个 Episode 独立落盘，支持 `--resume`；`--recompute` 只从现有 RoundEvent 重算统计，不调用模型。运行器为 `src/game_theory_agent/experiments/stage51_real_game_theory.py`。

## 3. 市场真实反事实口径

每轮保存真实 Joint Action 后，在完全相同的 `state_before`、Seed、其他公司动作和 MarketEnv 公式下，只替换公司 A 的价格，枚举价格边界、当前价、上下 500 分及 Advisor 候选价格。以真实下一状态的 Round Profit 最大值作为有限价格网格 Best-response Proxy：

- `Regret = max(0, 网格最佳真实利润 - 实际真实利润)`；
- Advisor Counterfactual 比较 Advisor 推荐价格与实际动作在同一环境中的真实利润；
- Enterprise Value、Share、Resilience 和 Incident Cost 也来自真实 MarketEnv 结果。

此外每次只改变一个对手的价格并枚举有限网格，报告最坏真实利润损失，命名为 `bounded_price_exploitability_proxy`。它不是连续动作、多 Agent 随机博弈的正式 Exploitability，也不声称求出均衡。

## 4. 工程验证与发现的缺陷

先使用 Mock 运行四组各 5 轮。首轮 C 组暴露 Replay 缺陷：Opponent Model 开启而 Utility 作为消融变量关闭时，Replay 把“预期没有 Utility 输出”误判为“Utility 输出丢失”。修复后 Utility、Advisor 和 Repeated Game Replay 都接收 Manifest，以显式 treatment mode 区分关闭和开启但数据缺失；第二个全新 Mock 目录四组全部通过。

真实 Pilot 后又收紧了战略引用统计。普通的“利润”“增长”等文字不再算作引用 Opponent Model；只有明确提到对手模型、策略分布、价格进攻性或具体策略类型才计数。这样避免用关键词误报“模型真正使用了战略输入”。市场配置路径也在导入 API 配置前读取，避免工作区移动或 `.env` 自定义路径导致市场配置和 Persona 配置混用。

最终全量后端测试为 181 项全部通过。真实四组的 Economic、Interaction、Information、Belief 和 GameTheory Replay 均为 100%，模型 fallback 为 0。

## 5. 真实模型单 Seed Pilot

配置：豆包 `doubao-seed-2-0-lite-260215`、Persona=`balanced_v1`、Seed=1001、每组 5 轮、temperature=0、top_p=1、公司 A 为 LLM、B/C/D 为 Rule。共 4 个 Episode、20 次真实决策调用。

| 组 | A 总利润（分） | 终局企业价值（分） | 终局份额 ppm | 平均一步 Regret（分） | 有限价格 Exploitability Proxy（分） |
|---|---:|---:|---:|---:|---:|
| A Persona Only | 6,342,734 | 41,667,799 | 193,088 | 102,396.0 | 798,920.6 |
| B + Belief | 6,775,726 | 42,492,459 | 208,906 | 103,534.0 | 843,233.6 |
| C + Opponent Model | 7,647,629 | 43,039,698 | 196,843 | 139,460.6 | 845,319.4 |
| D + Utility + Advisor | 1,803,089 | 36,076,225 | 294,112 | 896,698.6 | 707,810.6 |

本 Seed 的配对差值：

- B−A：利润 +432,992，企业价值 +824,660，份额 +15,818，但 Regret +1,138；
- C−B：利润 +871,903，企业价值 +547,239，份额 −12,063，Regret +35,926.6；
- D−C：利润 −5,844,540，企业价值 −6,963,473，份额 +97,269，Regret +757,238；
- D−A：利润 −4,539,645，企业价值 −5,591,574，份额 +101,024，Regret +794,302.6。

四组 Incident Cost 都为 0，因此本 Pilot 对 Risk Loss 没有辨识力。A/B/C/D 平均请求价格分别为 9,870、9,845.2、9,808 和 8,600 分；D 组 5 轮都明确引用 Advisor/Utility，并采用 Advisor 推荐价格。结果说明 Advisor 在本 Seed 中推动了低价抢份额，获得最高份额，却严重牺牲利润、企业价值和真实 Regret。其有限 Exploitability Proxy 下降 137,508.8 分，代表最坏单个对手价格响应下的短期损失范围缩小；这不能抵消真实利润恶化，也不能解释为整体战略更优。

B 组 5/5 轮明确引用概率信念。C 组虽然输入包含 Opponent Model，但 0/5 轮在 Planner Output 中明确引用策略类型或模型；因此 C−B 的差异只能称为 treatment 下的结果差异，不能据此断言真实 LLM 理解了 Utility 或 Opponent Type。D 组 5/5 引用 Advisor、5/5 引用 Utility、4/5 引用 Opponent Model。

## 6. Token 记录

真实 Pilot 共 20 次有 usage 的模型调用：输入 263,953 tokens，输出 12,229 tokens，总计 276,182 tokens，缺失 usage 的调用为 0。分组总 Token：A 60,577；B 63,666；C 69,504；D 82,435。D 的上下文最重，单 Seed 已明显增加成本。

## 7. 当前结论与证据等级

确定性工程证据：Evaluation 管线、处理隔离、真实 MarketEnv 一步反事实、五层 Replay、Token 汇总和中断恢复均已闭合。

方向性真实模型证据：Action Belief 和 Opponent treatment 在 Seed 1001 的累计利润/企业价值高于 Persona-only；Advisor treatment 明确改变了动作并在真实市场收益上显著变差，证明“内部 Proxy Regret=0”不能代表市场决策质量；真实模型可能读取 Belief，却可能忽略 Opponent Type。

不能得出的结论：一个 Seed 不能证明平均效应或显著性；C 组没有明确战略引用，不能证明 Utility Inference 比 Action Prediction 强；没有事故样本，不能比较 Risk；当前固定 Rule 对手不能回答混合类型、未知类型和 Holdout 泛化；没有 Cooperation/Repeated Game，不能回答合作、背叛和报复。

## 8. 下一步实验门槛

先保留本次负向 Advisor 结果，不修改冻结接口。下一批应按共同 Seed 扩展，并在正式 20-Seed 前先做 3–5 Seed 成本/稳定性检查；继续报告 B−A、C−B、D−C 和 D−A 配对差值。若 Advisor 的利润损害跨 Seed 重复，应检查其 Utility Proxy 标度与市场真实利润目标错配，而不是增加新 Advisor。

随后才执行：已知四类型、60/40 混合类型、随机未知权重和 Holdout 对手的 Generalization；再单独运行带 Cooperation/Repeated Game 的五人格实验。按当前 Pilot 的 Token 量级线性外推，完整 20 Seeds × 4 条件 × 20 轮约为 1,600 次调用和约 2,200 万 tokens，实际用量仍以供应商 usage 为准。

产物目录：`runs/stage51-real-pilot-doubao-seed1001-20260821/`。失败的首个 Mock 证据保留在 `runs/stage51-mock-smoke-20260821/`，修复后的 Mock 验收在 `runs/stage51-mock-smoke-v2-20260821/`。

## 9. 五个共同 Seed 稳定性扩展

用户要求先扩展到 3–5 个共同 Seed 检查 Advisor 的负向效果。实验沿用 Seed 1001 的权威产物，在同一目录增加 1002–1005；最终为 Seeds=`1001,1002,1003,1004,1005`、每组 5 轮，共 20 个 Episode、100 次真实决策。模型、Persona、Market、三个 Rule 对手和采样参数保持不变，四组顺序继续按 Seed 轮换。

全部 20 个 Episode 通过，模型 fallback=0，Economic、Interaction、Information、Belief、GameTheory Replay 全部 100%。五 Seed 的组均值为：

| 组 | 平均总利润（分） | 平均终局企业价值（分） | 平均终局份额 ppm | 平均一步 Regret（分） | 有限 Exploitability Proxy（分） |
|---|---:|---:|---:|---:|---:|
| A Persona Only | 8,610,631.4 | 44,626,096.0 | 237,328.2 | 767,978.2 | 849,688.3 |
| B + Belief | 9,075,150.0 | 44,798,200.4 | 228,836.8 | 470,894.3 | 839,005.2 |
| C + Opponent Model | 3,757,719.4 | 40,090,063.4 | 242,096.8 | 729,252.4 | 693,399.5 |
| D + Utility + Advisor | 2,268,770.2 | 36,724,876.0 | 288,389.2 | 1,640,327.3 | 654,099.6 |

### 9.1 Advisor 相对 Persona-only

D−A 的五 Seed 平均差值：利润 −6,341,861.2 分、企业价值 −7,901,220 分、份额 +51,061 ppm、Regret +872,349.12 分、有限 Exploitability Proxy −195,588.76 分。逐 Seed 方向完全一致：利润 5/5 降低，企业价值 5/5 降低，份额 5/5 提高，Regret 5/5 恶化，有限 Exploitability Proxy 5/5 降低。

对应双侧精确符号检验均为 `p=0.0625`。这是五个共同 Seed 下最强可能的双侧符号检验结果：说明方向高度稳定，但样本量仍不足以跨过传统 0.05 阈值，不能写成“统计显著”。研究结论应表述为“稳定的负向方向性证据”。

### 9.2 Advisor 相对直接上游 Opponent Model

D−C 平均利润 −1,488,949.2 分、企业价值 −3,365,187.4 分、份额 +46,292.4 ppm、Regret +911,074.84 分。Regret 在 5/5 Seeds 恶化；利润和企业价值在 4/5 Seeds 恶化。唯一例外 Seed 1004 中，C 组自身出现 −13,988,700 分巨亏，而 D 组为 +481,283 分，因此 D 相对 C 更好；但 D 仍远低于该 Seed 的 A 组 +10,689,594 分。这说明应同时报告 D−C 与 D−A，避免因为上游 treatment 自身崩溃而误判 Advisor 有效。

D 的 25 轮中 24 轮最终价格与 Advisor 推荐价格一致；平均请求价格 8,649.9 分，服务和韧性预算均为 0。Advisor/Utility 的显式引用分别是 20/25 和 24/25，但 Opponent Model 的显式引用只有 4/25，且都来自 Seed 1001。机制解释仍是：当前 Advice 稳定推动低价抢份额并压缩投入，获得更高 Share 和更低有限最坏价格响应损失，却牺牲真实 Profit、Enterprise Value 和一步 Regret。

### 9.3 Belief 与 Opponent Model

B−A 平均利润 +464,518.6 分，4/5 Seeds 为正；平均 Regret −297,083.84 分，但只有 3/5 Seeds 改善，企业价值只在 2/5 Seeds 更高。它是弱正向证据，不构成稳定结论。

C−B 平均利润 −5,317,430.6 分，只有 1/5 Seed 为正，且均值受 Seed 1004 的 C 组巨亏明显影响。C 组在 25 轮中对 Opponent Model/策略类型的明确引用为 0。因此当前数据不支持“Opponent Model 比 Action Prediction 更强”；更合理的新猜想是 Opponent Model 虽进入 Context，但没有被真实模型转换为可区分的动作原则。

### 9.4 Token、费用和下一门槛

五 Seed 总 usage：输入 1,317,130、输出 61,518、总计 1,378,648 Tokens，100/100 调用均有 usage；单次最大输入 20,585 Tokens，全部属于豆包 2.0 Lite 的 ≤32K 档。按现金价输入 0.0006 元/千 Token、输出 0.0036 元/千 Token，估算总费用约 1.0117 元；相对首个 Seed 新增约 0.8093 元。

五 Seed 已足以确认需要认真对待 Advisor 的目标错配，但仍不修改冻结接口。下一门槛应增加到至少 10 个共同 Seed：如果 D−A 继续保持利润/企业价值全负、份额/Regret 全正，双侧符号检验将降至 `p=0.001953125`；然后再判断是校准 Utility Proxy，还是修改 Advisor 对份额、价格战和商业投入的权重。Generalization 与 Repeated Game 仍应作为独立实验，不能混入这组消融。
