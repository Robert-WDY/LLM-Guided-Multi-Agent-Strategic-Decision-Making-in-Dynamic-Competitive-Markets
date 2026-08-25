# Strategic Research MVP v0.7：结论与边界

本发布是中期研究快照，不是最终产品，也不宣称所有博弈模块都能提高收益。

## 可以展示的工程结论

- MarketEnv 支持 2–10 公司、同时行动、随机事件、事故、公共/私有观察、通信、共享抗冲击投入和确定性结算。
- Economic、Interaction、Information、Belief、Advisor 与 Adoption 记录可从冻结 RoundEvent 重建；本发布三个选中档案的对应回放全部通过，隐藏状态泄漏数为 0。
- Proposal、Acceptance、Commitment、Actual Contribution、Partial Betrayal、Credibility Update 和 Public Benefit 的合作链路已经实现。
- Agent Version 可绑定 Prompt、Persona、Strategic Stack、源码、依赖和 Provider 配置；真实输出按 Tier C 保存并归属。

## 有配对证据的研究结果

- 人格异质性在 10 个共同 Seed 中 10/10 扩大价格差、显著降低定价趋同，但市场利润在 9/10 Seed 下降，且没有解决集体投资不足。异质性打破趋同不等于提高福利。
- v6 逐项和组合边际候选在五个已知失败状态中 5/5 相对真实 LLM Control 提高企业价值，平均增加 453,268 分；这是已知状态修复证据，不是未知状态泛化结论。
- 合作机制的确定性市场实验能够产生共同投入、搭便车、部分背叛和集体行动失败。

## 本轮明确失败的门禁

零 Token Forecast 校准的独立历史 Holdout 共 90 条样本：全体方向一致率 91.11%，放行精度 94.12%，Coverage 62.96%；但 17 条放行候选中仍有 1 条权威边际为负。因此 `zero_token_holdout_gate_passed=false`，真实 v6 最多 6 次复测没有运行，新增模型调用和 Token 均为 0。

## 不能宣称的内容

- 不能宣称 v6 已经在未知状态或新模型请求上稳定提高长期企业价值。
- 不能宣称 Belief 或 Opponent Model 已经稳定降低 Regret；当前 Belief 预测仍弱。
- 不能宣称已经实现并验证 Cooperator、Free Rider 或 Retaliator 数学人格。合作机制存在，但后端 Persona Catalog 的合作 Capability 仍关闭。
- 不能宣称市场参数具有现实经验效度。价格弹性、投入回报、事故概率和终值仍是版本化合成参数。
- 尚未开始 Self-play、PSRO、策略训练、现实数据校准或多用户平台。
- 真实模型 temperature=0 也不保证重新生成相同文本；可重建的是已保存 Raw Output 后的解析、约束、动作和市场结算。

## 证据等级

- `ENGINEERING`：证明合同、状态转移、信息边界或回放正确。
- `DIRECTIONAL`：小样本、配对 Smoke 或已知失败集，只支持方向性判断。
- `REAL`：档案包含真实模型输出；单 Episode 仍不构成统计结论。
