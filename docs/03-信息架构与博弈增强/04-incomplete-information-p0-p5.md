# 不完全信息 P0–P5 实现与验收

## 1. 范围

本阶段一次性完成最小、可审计的五层链路：

```text
True MarketState
  -> strict PublicState / PrivateState / ObservationEnvelope       (P0)
  -> deterministic public-action Belief                            (P1)
  -> next price cut / maintain / raise prediction                  (P2)
  -> Belief + Advice enter Planner                                 (P3)
  -> visible non-binding communication signal + reliability        (P4)
  -> approximate Bayesian price-response advisor                   (P5)
  -> independent final Agent action
  -> MarketEnv settlement
```

它没有修改 MarketEnv 需求、利润、风险或合作公式。消息、信念和 Advisor 都不能直接执行动作。

## 2. P0：严格信息契约

新增一等 Pydantic 契约：

- `PublicState`：同轮所有公司完全一致；
- `PrivateState`：完整状态只能属于观察公司；
- `ObservationEnvelope`：`extra=forbid` 的 Gateway 顶层契约；
- `ObservationSnapshot`：继续绑定 True State、公司、Policy、Belief 与 Observation Hash。

经济子文档继续使用自身市场协议版本，Envelope 负责阻止未登记顶层字段绕过可见性策略。`ObservationBuilder` 仍是唯一 True State→View 入口，Information Replay 仍从权威 True State 重新投影，而不是信任日志视图。

## 3. P1/P2/P3：信念、预测与 Planner

`public_action_v1` 保持完全兼容：只使用已结算公开价格，Dirichlet(1,1,1) 形成下一轮降价、持平、涨价概率。

`DecisionContext v1.13` 与 `market-planner-prompt-v1.13` 接收 Belief 和可选 `game_theory_advice`。Prompt 明确：概率不是事实，Advisor 不是指令。此前真实模型 36 次调用的 Pilot 证据仍成立；本阶段没有新增付费真实模型调用。

## 4. P4：Communication Signaling

新增 treatment `belief_mode=public_action_signal_v2`，要求通信开启。只解析实际可见消息中的结构化 `own_action_claim.price_cents`：

- 私信只进入发送者的目标收件人信念；
- 非目标公司不知道 Signal ID、数量或正文；
- 观察者不会把自己的声明建模为“对手信念”；
- 自由文本不会被主观解析为事实；
- Signal 固定标记 `verified_fact=false`、`non_binding=true`；
- 当前声明按发送者历史结构化声明可靠度加权。

可靠度使用 Beta(1,1) 后验均值。首次声明前为 50%；结构化价格声明与最终动作不一致后变为 33.3333%；一次真、一次假后回到 50%。Settlement Hash 同时绑定公开价格和结构化声明，重复结算不能换一套信号。

这仍不是“听见一句话就改变事实”。信号只改变 company-scoped Belief，实际市场只由最终 Joint Action 改变。

## 5. P5：Bayesian Game Advisor

新增 `advisor_mode=bayesian_price_v1` 和独立 `advisor/` 包。Advisor：

1. 读取本公司私有当前价格、单位成本和合法报价边界；
2. 对每个对手的公开方向概率求期望竞争压力；
3. 在有限候选报价上计算透明的 Expected/Downside Payoff Proxy；
4. 输出非绑定推荐、候选评分、限制说明和 Advice Hash；
5. 由 `verify_advisor_replay` 从记录的 Belief 与公司输入重建。

它不会读取对手现金、成本、Persona、效用或隐藏计划，也不会输出 `final_action`。这是 Approximate Bayesian Price Response，不是精确 Nash/Bayesian Nash 求解器；代理仍可依据 Persona、现金和其他经营目标拒绝建议。

## 6. 验收结果

命令：

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.experiments.incomplete_information_acceptance `
  --output runs/incomplete-information-p0-p5-acceptance-20260821
```

验收产物：`runs/incomplete-information-p0-p5-acceptance-20260821/summary.json`。

固定 Seed `20260821` 的结果：

- 17/17 工程检查通过；
- 对手私有字段泄漏数 0；
- PublicState 完全相同，PrivateState 归属正确；
- A→B 私密降价声明只进入 B 的对手信念，C/D 信号数为 0；
- B 对 A 的降价概率从中性 `333334 ppm` 上升到 `428572 ppm`，C 保持 `333334 ppm`；
- 该未履行声明结算后，A 的声明可靠度从 `500000 ppm` 降到 `333333 ppm`；
- Communication Close 前后 State Hash 完全相同：`sha256:dd8d3ac9a76cfc854233e9bc47c91a53852e20ac2e5191c6a80b917614aa89b4`；
- Economic、Interaction、Information、Belief、Advisor Replay 全部 100%；
- 全量后端测试 173 项通过。

## 7. 实现中发现的问题

### 离线实验 Observation 不完整

强类型 Envelope 首次接入后，Persona 离线实验缺少 `last_settled_round`、`episode_config` 和 `company_analysis`，全量回归失败。这说明此前 API 与离线实验虽然共用 Builder，外层 Envelope 仍存在形状漂移。修复为离线实验补齐相同绑定字段，随后全量回归恢复。

### 发送者自己的 Signal 不应进入 Opponent Belief

首次验收脚本错误地要求 A 的私信声明同时出现在 A 的 Belief。CommunicationView 中 A 确实可见自己发送的私信，但 BeliefState 明确只建模对手，因此 A 的 `visible_communication_signals` 应为空。修正验收口径后通过；这不是隐私泄漏或消息丢失。

### 版本兼容

Planner 新增 Advisor 语义，因此 Prompt 从 v1.12 升到 v1.13，DecisionContext 从 v1.12 升到 v1.13，Manifest 升到 v1.5。旧 Context 版本仍可解析，默认 belief/advisor 均为 off，不改变旧市场路径。

## 8. 结论边界

当前已经证明：P0–P5 工程链路可运行、隔离正确、可重建、信号能够以可靠度加权改变目标公司的动作预测，Advisor 能把同一 Belief 转换为确定性策略建议。

当前没有证明：真实 LLM 会稳定采纳 Signal/Advisor、预测精度足够高、Advisor 提高利润、系统收敛到 Bayesian Nash Equilibrium，或通信能够产生稳定合作。上述问题仍需要共同 Seed、多次真实模型调用、placebo 和策略切换实验。
