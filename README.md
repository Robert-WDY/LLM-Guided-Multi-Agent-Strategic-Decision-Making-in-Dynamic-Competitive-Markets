# LLM-Guided Multi-Agent Strategic Decision-Making in Dynamic Competitive Markets

本项目研究 LLM 智能体在竞争市场中如何做策略决策：它们的价格和利润离博弈论基准（一次性 Nash 均衡 / 联合垄断）有多远，人格、公开通信和博弈论顾问工具又如何改变市场结果与社会福利。

## 当前主张（v21，2026-11）

<!-- RESULTS-START -->
结果见 [docs/oligopoly-v1-results.md](docs/oligopoly-v1-results.md)。
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
