# Stage 6.9 Advisor 弃权修复：零 Token 回归结果

## 结论

`pareto_reliable_v7` 已把“不确定时执行 fallback”改为真正弃权：`execution_disposition=defer_to_agent`、`recommended_action=null`，Agent 保留自己独立生成并通过硬约束处理的动作。

在 Stage 6.8 已知的 5 个负面窗口中：

- v7 弃权：5/5；
- 弃权后仍产生可执行建议：0/5；
- 因弃权产生的相对 Agent 原计划负增量：0/5；
- v6 fallback 在这些窗口造成的历史损失合计：2,230,133 分（22,301.33 元）；
- 新真实模型调用、Token 与成本：均为 0。

结果数据见 `summary.json`，内容 Hash 为 `sha256:587d155c91851a9e57868bb799ce37fe5614b422d0c5204e0e7ade628b54d64d`。

## 工程含义

本次回归证明的是安全语义已经修正：当可靠性门禁拒绝 Planner 建议时，系统不再把 `status_quo` 或 `risk_buffer` 包装成安全动作，也不会把 withheld candidate 交给 Agent 执行。采纳追踪会把这种情况记录为 `unavailable`，而不是接受、部分接受或拒绝某个实际建议。

v6 的契约、Hash 和历史 Replay 保持不变；v7 使用独立 Advice、Gate 和 Adoption Trace 版本。

## 研究边界

这 5 个窗口已经用于 Stage 6.8 诊断，因此只能作为工程回归，不能作为新的未知样本证据。这里的“负增量为 0”表示 v7 弃权后没有覆盖已记录的 Agent 动作，不代表 v7 已经在新的真实 LLM Episode 中提高长期收益。

在重新打开真实模型付费门禁前，还需要使用全新 Seed 做零 Token 留出测试；通过后最多先进行 3 个新 Seed 的成对真实模型 Smoke。
