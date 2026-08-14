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
- 随机 / 固定 Seed 前端入口和 200 Seed 投入档位校准。
- 规则对手使用 Seed 固定的价值型、溢价型、增长型或谨慎型行为，并带可复现的回合扰动；不调用 Agent 或模型；
- 利润计入固定运营和逐单履约成本，声誉采用慢变量更新，完全维修当轮保留残余事故影响；
- Agent 公开观察包含逐轮历史、预警兑现结果、成本与事件传导解释，以及终局企业价值排名。
- 可选择均衡、价格敏感、品质偏好、服务偏好或 Seed 随机市场；市场公开需求偏差、价格锚点与合理价格区间。
- 品质市场显著强化品牌知名度与声誉，服务市场显著强化当期服务与历史缺货惩罚，不再仅靠消费者分群比例形成细微差别。
- Episode 支持 `5 / 10 / 15 / 20` 轮、随机或固定 uint64 Seed，并提供 Agent 只读选项发现接口；创建与结算仍由私有 Controller 负责。
- 对手报价围绕市场锚点、上一轮成交价和成本底线重新计算，不再把自身上一轮报价当作只能上涨的硬下限。
- 终局同时提供综合价值榜和总资产榜、四家公司逐轮综合价值折线，以及与冠军逐项对比的结果解释。
- `settled_market` 与 `public_history[].market` 使用同轮事件、情绪和供应成本口径；顶层 `state.market` 表示下一决策轮当前条件。

前端已经接入后端 API，不保留独立市场公式，并同时支持单公司经营和市场全景。

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
市场模型、榜单口径与回溯字段见 [docs/market-models-and-ranking.md](docs/market-models-and-ranking.md)。
四人协作的分支、Review、模块边界和合并检查见 [CONTRIBUTING.md](CONTRIBUTING.md)。每次 Pull Request 会自动运行后端测试、前端 lint 和生产构建。

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
