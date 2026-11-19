# Oligopoly v1 实验计划：LLM 定价智能体离 Nash 有多远

本计划在看到主实验数据之前提交到 git（提交时间即时间戳；**这不是第三方预注册**，只用于防止事后改假设）。试点数据（`experiment-results/oligopoly-v1/pilot/`）只用于定轮数、样本量，不进入主分析。

## 研究问题

- **RQ1**：不同厂商、不同规模的 LLM 在重复定价博弈中，价格和利润离一次性 Nash 均衡与联合垄断各有多远？
- **RQ2**：人格、公开通信（cheap talk）、博弈论顾问（最优反应 / Nash 计算工具）如何改变这一距离和社会福利？
- **RQ3**：卖家数量（2 vs 4）和对手类型（LLM / 规则对手）的影响。

## 市场

Calvano et al.（2020）的 logit-Bertrand 市场：a=2, c=1, a0=0, μ=0.25。2 家时 p_N=1.4729、p_M=1.9250。按 Fish, Gonczarowski & Shorrer（2024），每个种子的价格尺度 α∈{1, 3.2, 10}（seed mod 3），避免模型背诵教科书价格。智能体只看到自己的价格、销量、利润和对手报价，不知道需求函数，也不知道总轮数。

## 指标

- **主指标** Δ = (π̄ − π_N)/(π_M − π_N)，按最后 15 轮的平均每家利润计算（Calvano 合谋指数）。
- 次要指标：价格指数 (p̄ − p_N)/(p_M − p_N)、总福利、消费者剩余损失（相对 Nash，按 Nash 与垄断之差归一化）、无效回复率。
- 参照：Calvano Q-learning 16 次会话 Δ = 0.836（SD 0.121）；近视最优反应对手收敛到 Δ = 0。

## 设计

- 每局 50 轮，最后 15 轮记分（试点显示约 15–20 轮内锁定）。思考模式关闭，温度用厂商默认值。
- 每格 20 个独立种子（n=4 时每格 12 个）。试点 SD ≈ 0.45 时，20 vs 20 的双侧 5%、80% 功效最小可检测差为 Δ ≈ 0.40；n=4 每格 12 个时约为 0.52。效应小于这些值的结果报告为"未检出"，不加种子追显著。
- 模型：db-mini（doubao-seed-2-0-mini）、db-lite（doubao-seed-2-1-lite）、db-pro（doubao-seed-2-1-pro）、ds-flash（DeepSeek-V4.1-Flash）、ds-pro（DeepSeek-V4-Pro）。

| 实验 | 格 | 说明 |
|---|---|---|
| E1 规模 | 5 个模型 × base（中性人格、无通信、无顾问） | RQ1 |
| E2 机制 | {ds-flash, db-lite} × {comm, advisor-br, advisor-full, aggressive, cooperative} | 与同模型 E1 base 对比 |
| E3 结构 | {ds-flash, db-lite} × n=4 × {base, comm}；ds-flash vs 近视最优反应；ds-flash vs db-lite 异质对局 | RQ3 |

## 预先设定的假设（Holm 校正，族 = H2–H6 共 6 个检验）

- **H1**：各模型 base 的 Δ > 0（单样本，Bonferroni 5 个模型）。
- **H2**：通信提高 Δ（每个模型一次检验，合并时用分层置换）。
- **H3**：advisor-br（只给最优反应和 Nash）降低价格指数。
- **H4**：advisor-full（额外给出联合最优价）提高 Δ。
- **H5**：aggressive 降低 Δ，cooperative 提高 Δ（两个检验）。
- **H6**：n=4 的 Δ 低于 n=2。

检验：格间差用 10,000 次置换的双侧置换检验，95% 置信区间用 10,000 次 bootstrap。所有格的全部结果都报告，包括不显著的。

## 预算

所有调用 token 均有记录，硬上限 45M token；超出即停，并如实报告完成到哪一步。
