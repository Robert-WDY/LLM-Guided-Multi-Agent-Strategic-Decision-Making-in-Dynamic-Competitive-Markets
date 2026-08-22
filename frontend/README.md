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

高级 Episode（通信、合作、Belief、Opponent Model、Utility 或 Advisor）受 Controller Token 保护。Token 由用户在本地页面输入，只保存在 React 内存中，不写入 URL、Local Storage 或演示数据。前端只能创建 Episode；真实高级回合必须由 Coordinator 完成 Communication Close、Agent Intent 和 Settlement，前端不会绕过屏障直接调用市场 Step。

后端不可用或未提供 Token 时，可以载入 Research Demo，检查全部页面与交互。演示严格从第 1 回合开始，到最大回合后结束，不会从 20 循环回 1。演示市场用一个确定性的 UI-only 竞争函数让价格与投入影响订单吸引力，并在每轮将四家公司份额归一化为 100%；它只用于验证交互和解释因果，不复制也不替代 Python `MarketEnv`，不产生研究结论。真实结果必须来自后端 RoundEvent 与 Replay。

## 验收

```powershell
npm run lint
npm run build
npm test
```

视觉验收还应覆盖桌面宽屏与窄屏，检查八个导航页、Persona Drawer、Agent 切换、Observation Tabs、通信过滤和 Replay 节点选择。实时页需额外验证：初始 Round=1、最大回合不可循环、每轮 Share Total=100.0%、至少一家公司跨轮既有上升也有下降，以及不同 Human Action 会得到不同市场结果。
