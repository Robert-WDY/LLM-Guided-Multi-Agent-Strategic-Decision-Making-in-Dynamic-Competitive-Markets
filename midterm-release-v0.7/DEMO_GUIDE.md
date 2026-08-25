# 中期答辩演示指南

## 启动

在项目根目录运行：

```powershell
.\midterm-release-v0.7\scripts\start_midterm_demo.ps1
```

脚本检查 Python、Node、依赖目录和冻结配置，隐藏启动后端与前端，完成健康检查后打开 `http://localhost:3210`。演示不需要 API Key，也不会调用真实 LLM。

## 推荐的 12 分钟演示顺序

1. 从首页进入“研究控制台”，说明观察、个人体验和研究入口彼此分开。
2. 打开“真实实验档案”，选择“人格异质性真实市场样例”。强调 `REAL`、只读和单 Seed 结论边界。
3. 选择第 3 回合、Company B，展示 Observation Hash、当时可见信息、最终动作和 Replay Hash。
4. 切换“权威审计视角”，说明完整状态必须使用本地 Controller Token；不提供 Token 时必须拒绝。
5. 打开“合作机制验收”，展示 Proposal → Acceptance → Commitment → 30% 履约 → Partial Betrayal → Credibility Update。明确它是 `ENGINEERING` 证据，不是合作人格实验。
6. 打开“旧 Bayesian Advisor 真实决策”，展示 Advice、Agent 采纳记录和回放。
7. 展示 `selected-runs/forecast-calibration/summary.json`：总体精度通过但仍有一个负放行，所以没有花费 6 次真实调用。
8. 以人格异质性的收益与代价作为中期主要研究结论收尾。

## 演示口径

- “合作机制已实现”可以说；“合作人格已完成”不能说。
- “v6 修复五个已知失败状态”可以说；“v6 已证明泛化有效”不能说。
- “选中真实档案可完整回放”可以说；“模型能重新生成同一句话”不能说。
- 演示档案是冻结证据，页面不会启动付费高级实验。
