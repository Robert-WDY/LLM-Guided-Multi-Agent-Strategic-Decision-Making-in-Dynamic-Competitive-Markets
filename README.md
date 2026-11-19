# LLM-Guided Multi-Agent Strategic Decision-Making in Dynamic Competitive Markets

本项目研究 LLM 智能体在竞争市场中如何做策略决策：它们的价格和利润离博弈论基准（一次性 Nash 均衡 / 联合垄断）有多远，人格、公开通信和博弈论顾问工具又如何改变市场结果与社会福利。

## 当前主张（v21，2026-11）

<!-- RESULTS-START -->
在 Calvano et al.（2020）的重复 logit-Bertrand 定价博弈中，让 DeepSeek 和豆包的 5 个模型（共 606 局主实验）在不知道需求函数的情况下互相定价：

- **4/5 个模型把价格定在 Nash 之上**，合谋指数 Δ 为 0.38–0.70（0 为 Nash，1 为联合垄断；Q-learning 基准 0.84）。豆包系列内 Δ 随模型规模上升，最小模型约为 0。
- **50 轮内一半以上来自开局锚定**：把第 1 轮固定在 Nash 价后，Δ 降到 0.21–0.29；时间拉到 150 轮后，只有 DeepSeek-Flash 继续缓慢上漂（0.30 → 0.44），另外两个模型没有。
- **博弈论顾问的作用最大**：只给出"最优反应 + Nash 价"的顾问让价格几乎完全回到 Nash（价格指数 0.02）；只有一家装这个顾问时，它定价更低、赚得比对手多，市场 Δ 减半；但和自己不装时相比，没有证据表明它赚得更多（DeepSeek Flash 反而更少）。额外给出联合最优价时，结果因模型而异。
- **通信和"合作型"人格不提高利润，却让价格超过垄断价**，总福利损失为垄断的 1.2–2.4 倍；"激进"人格把 Δ 压到 0 附近。
- 4 家卖家时 Δ 不比 2 家低；打开推理模式、换一种 prompt 措辞都不显著改变 Δ（但措辞会影响价格水平）。面对近视最优反应的规则对手时，LLM 自己维持高价，利润被对手赚走。

Final report（PDF）：见仓库的 [Releases](https://github.com/Robert-WDY/LLM-Guided-Multi-Agent-Strategic-Decision-Making-in-Dynamic-Competitive-Markets/releases)；完整结果、检验与局限：[docs/oligopoly-v1-results.md](docs/oligopoly-v1-results.md)；实验计划：[experiment-specs/oligopoly-v1/PLAN.md](experiment-specs/oligopoly-v1/PLAN.md)。
<!-- RESULTS-END -->

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/game_theory_agent/oligopoly/` | **当前主线**：logit-Bertrand 重复定价博弈、Nash/垄断基准、LLM 定价智能体、Q-learning 基线 |
| `scripts/oligopoly/` | 实验网格（`grids.py`）、运行器（`run_study.py`）、分析（`analyze.py`） |
| `experiment-specs/oligopoly-v1/PLAN.md` | 看到主实验数据之前提交的实验计划和假设 |
| `experiment-results/oligopoly-v1/` | 每局摘要、逐轮轨迹（`.jsonl.gz`）、模型回复缓存（可离线复算） |
| `src/game_theory_agent/`（其他子包） | v4–v20 的生鲜配送市场引擎、Agent Gateway、建议器等历史系统，测试仍全部保留 |
| `frontend/` | v10–v20 研究控制台（React） |
| `docs/INDEX.md` | 全部文档的分类索引 |

## 复现

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test,analysis]'
.venv/bin/python -m pytest                     # 全部测试

# 离线复算：只读已归档的模型回复，不发出任何 API 请求
.venv/bin/python scripts/oligopoly/run_study.py --phase e1 --offline
.venv/bin/python scripts/oligopoly/analyze.py  # 生成表格、检验和图

# Q-learning 基线（Calvano et al. 2020）
.venv/bin/python scripts/oligopoly/qlearning_baseline.py 16 2
```

重新调用模型需要在 `.env` 中设置 `DEEPSEEK_API_KEY` 和 `ARK_API_KEY`（不要提交）。已完成的局会自动跳过；同一请求命中缓存后不会重复计费。

## 历史

v4–v20（生鲜配送市场、可靠建议器）的版本日志见 [docs/archive/README-history-v4-v20.md](docs/archive/README-history-v4-v20.md)；2026-11-18/19 的续约否决研究线已结项，见 [docs/renewal-veto-closeout.md](docs/renewal-veto-closeout.md)。

MIT License.
