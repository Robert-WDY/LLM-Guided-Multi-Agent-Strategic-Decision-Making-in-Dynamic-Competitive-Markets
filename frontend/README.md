# Game Theory Lab 前端

这是 LLM 多智能体博弈实验平台，不是单纯的市场游戏页面。首页先提供三个清晰入口：

- 个人体验：作为 Human Agent 参与决策；
- 观察实验：不参与决策，只查看回合、消息、Agent 状态和市场结果；
- 研究控制台：使用完整的 Belief、Strategy、Replay 与实验工具。

每个入口都先进入独立的环境配置页，再进入对应工作区；配置页和运行页都可以返回主页。研究控制台内部保留以下工作区：

- Experiment Setup：市场、共同 Seed、信息条件、通信、合作、Game Theory treatment 与 Agent 构成；
- Live Simulation：从第 1 回合开始的同页决策舱，集中展示 Observation、Communication、Belief、Advisor、Human Action、公司竞争和 Joint Settlement；
- Agent Observatory：Agent-scoped Observation、Belief、Planning 与结构化 Decision Summary；
- Communication Center：公开/私信可见性和 Proposal→Commitment→Fulfillment；
- Belief & Strategy：Opponent Model、Utility Inference、Advisor 候选及 Final Action 差异；
- Market Dashboard：利润、份额和投入结构；
- Replay Viewer：True State→Observation→Belief→Communication→Action→Result；
- Experiment Report：工程证据、方向性证据和统计结论分层。

界面文案以中文为主，不展示模型隐藏思维过程。通信内容明确标为未验证、非绑定输入；演示数据明确标为交互演示，不冒充真实模型运行。实验报告在当前实验完成前不会出现在导航中，完成后生成的也是当前会话总结，不会混入历史实验结论。

实时页提供六步处理过程：接收信息、交流、形成判断、制定策略、做出决策、市场结算。可以切换公司，查看每个智能体在各步骤合法可见的上下文和结构化决策依据。第一回合没有历史行为，所以对手判断显示“未知”，不会伪造精确概率；第一轮结算后，才使用公开价格和份额变化更新判断。本轮消息和对手判断摘要始终可见，不需要展开面板。

`01 实时现场` 只提供紧凑 AI 进度；完整 PersonaAgent/SABM 八节点纵向拓扑位于 `02 智能体观察`。参与者模式的三家 AI 按三列并排，每列内部纵向展示八节点，不会把人类公司的“接收信息”误作模型节点事件。创建 Episode 后、尚未提交回合时所有节点明确显示“尚未运行”；真实回合完成后，01–08 节点分别显示真实记忆视图、实际决策输入、策略反思、模型提示词与三候选、选中理由、Token/耗时、校验/修复、待提交意图、提交回执和终态。

真实回合 POST 等待期间，前端每 500ms 读取后端安全进度快照。`01` 显示当前节点、模型等待、调用次数与实际已用时间；`02` 将最近进入的真实节点标为“执行中”，较早节点标为“已执行”。Provider 返回后才显示真实 Token 和模型耗时；不显示预计百分比、预计剩余时间、未完成 JSON 或模型原始输出。切换导航不会中断轮询，POST 完成后用完整终态 trace 替换临时事件。

参与者、观察者和研究员三种真实入口统一使用隔离的后端投影。模型运行前不会生成或预填消息、信念、策略、动作、结算、图表和回放；`03` 至 `06` 仅展示本次 Episode 实际产生的通信、策略 trace、市场结果与 history。显式交互演示仍使用独立演示状态，两者不混合。

## 本地运行

先从仓库根目录启动后端：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.api
```

再启动前端：

```powershell
cd frontend
npm ci
npm run dev
```

默认前端端口为 `3210`，市场 API 为 `http://localhost:8010/api`。可通过 `NEXT_PUBLIC_MARKET_API_URL` 修改。

API Key 固定从后端 `secrets/open_router-api_key.env` 读取，前端不提供密钥输入框，也不会接收、保存或发送密钥。创建、配置和验证步骤见 [OpenRouter API Key 创建与本地配置](../docs/member-2/05-openrouter-api-key-setup.md)。正常 UI 只显示“AI 模型”，仅缺密钥或 AI 调用错误时显示具体 Provider 排障信息。基础 Episode 通过后端管理接口运行模型 Intent、人类 Intent 和联合结算；通信、合作、Belief、Opponent Model、Utility 或 Advisor 等真实高级回合仍由本地 Coordinator 完成屏障协议。

新实验默认 5 回合。模型下拉框只展示真实结构化门控通过的固定免费模型：默认 Super，以及较慢的复杂推理备选 Ultra。

后端不可用时，可以载入 Research Demo，检查全部页面与交互。演示严格从第 1 回合开始，到最大回合后结束，不会循环回第 1 回合。演示市场用一个确定性的 UI-only 竞争函数让价格与投入影响订单吸引力，并在每轮将四家公司份额归一化为 100%；它只用于验证交互和解释因果，不复制也不替代 Python `MarketEnv`，不产生研究结论。真实结果必须来自后端 RoundEvent 与 Replay。

## 验收

```powershell
npm run lint
npm run build
npm test
```

视觉验收还应覆盖桌面宽屏与窄屏，逐一进入参与者、观察者和研究员真实模式，检查初始态无预填结果、`01` 紧凑进度、`02` SABM 纵向详情、`03` 实际通信、`04` 实际策略 trace、`05` Controller 经营结果和 `06` Episode history。实时页还需验证：初始 Round=1、最大回合不可循环、每轮 Share Total=100.0%，以及不同 Human Action 会得到不同市场结果。
