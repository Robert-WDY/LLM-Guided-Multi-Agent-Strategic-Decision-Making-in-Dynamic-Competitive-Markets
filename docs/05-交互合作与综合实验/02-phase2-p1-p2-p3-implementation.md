# 阶段 2 P1/P2/P3 实现报告

- 日期：2026-08-20
- 状态：工程实现完成，Mock 端到端验收通过；2 LLM + 2 Rule 真实模型小型 Smoke 已执行。

## 1. 本次交付口径

按本轮实施顺序重新编号：

- P1：通信 API、身份隔离、提交与 Communication Close；
- P2：Agent 独立通信调用、关闭后决策和协调器两段屏障；
- P3：RoundEvent 审计、Interaction Replay、指标和 20 轮验收。

通信仍然是非约束 Cheap Talk。合作人格、共同效用、合同执行、转账、联合投资和市场公式均未加入。

## 2. P1：通信协议与 API

Episode 新增：

```json
{
  "communication_mode": "off | public_only | public_private"
}
```

默认值为 `off`。通信开启时：

- 创建 Episode 必须使用 `X-Controller-Token`；
- 创建响应一次性返回每家公司的 `agent_tokens`；
- Session 只保存 Token 的 SHA-256，不保存明文；
- Observation、通信提交和 Intent 都校验 `X-Agent-Token`；
- Token 与 company 绑定，不能冒充其他公司或读取其私信；
- Intent 必须携带当前公司关闭视图的 `communication_view_digest`。

新增接口：

```text
POST /v1/episodes/{episode}/companies/{company}/communication/submissions
GET  /v1/episodes/{episode}/companies/{company}/communication/view
POST /api/v1/controller/episodes/{episode}/communication/close
```

每轮账本绑定：

```text
episode_id + round + state_version + state_hash
```

相同提交和 Close 幂等；替换已接受提交、过期状态、非法收件人和 Close 后提交均被拒绝。通信前后会再次比较市场状态 Hash，消息不进入 `MarketState`。

## 3. P2：Agent 两段调用

每轮真实执行顺序为：

```text
获取冻结 Observation
→ 运行 AgentRuntime.communicate()
→ 并发提交消息
→ Controller Communication Close
→ 重新获取各公司关闭后的 Observation
→ 运行 AgentRuntime.decide()
→ 提交 Intent
→ Action Lock 和唯一一次 MarketEnv.step()
```

新增 Agent 契约：

- `CommunicationContext`；
- `AgentCommunicationResult`；
- `CommunicationSubmission`；
- `DecisionContext.communication_view`；
- `AgentDecision.message_responses`。

通信生成失败、超时、非法 JSON 或客户端不支持通信时，结果转化为带错误原因的空 Submission，也就是可审计沉默。它不会阻止后续经营决策。

DeepSeek、豆包、Mock 和 Uniform Random 均已接入：

- DeepSeek/豆包通信输出支持一次 JSON 修复；
- Mock 可配置公开或私密消息，并能确定性接受指定价格建议；
- Uniform Random 主动沉默；
- `off` 模式兼容只有 `generate_decision()` 的旧客户端。

决策 Prompt 把对手消息放在：

```text
[UNTRUSTED_NON_BINDING_OPPONENT_MESSAGES_JSON]
...
[/UNTRUSTED_NON_BINDING_OPPONENT_MESSAGES_JSON]
```

消息块会转义保留分隔符，并明确标记为不可信、非约束数据，以降低 Prompt Injection 风险；真实模型的抵抗能力仍需单独 Smoke。无论消息内容如何，Runtime/Controller 都会继续执行可见消息引用校验和经济动作护栏。

## 4. P3：审计和双 Replay

`RoundEvent` 升级为 `agent-round-event-v1.4.0`，仍可读取 v1.0–v1.3。v1.4 明确记录并校验最近三轮、按公司权限过滤的通信历史。

每轮新增：

- 完整 `communication_phase`；
- Controller-only Closure 和 transcript hash；
- 每家公司 view digest、可见消息 ID 和自身消息 ID；
- 通信生成 Context、原始输出、合法 Submission；
- 模型、Prompt、耗时、Token、错误和沉默原因；
- 决策阶段实际使用的 `communication_view`；
- Agent 对消息的接受、拒绝或忽略记录。

Replay 分成两层：

1. `verify_replay()`：继续只根据最终联合动作重放经济市场；
2. `verify_interaction_replay()`：重新运行确定性通信账本，重建消息 ID、可见性、transcript hash 和各公司 view digest。

Interaction Replay 可以发现日志中的非一致性修改：

- 消息正文、发送者或收件人被篡改；
- 消息顺序、message ID 或 transcript hash 被篡改；
- DecisionContext 使用了错误视图；
- Agent 回应了自己无权查看的私信。

这些 SHA-256 是一致性 Hash，不是数字签名。若攻击者能同时重写事件全部字段并重算所有 Hash，仍需外部签名根 Hash 或事件 Hash 链才能提供真正的防篡改真实性。

新增 Interaction 指标包括消息数量、频道、speech act、沉默原因、通信 Token/延迟、消息回应，以及结构化行动声明与最终动作的精确 alignment/deviation。这些行为指标不参与工程 `passed` 判定。

## 5. 验收结果

### 自动化测试

- 全量测试：135 项全部通过；
- Python `compileall`：通过；
- `git diff --check`：通过。

覆盖的关键场景：

- Token 冒充和跨公司读取被拒绝；
- 4×4 全部有向私信可见性无泄露；
- 返回对象被修改也不能污染权威私信账本；
- 并发提交到达顺序不改变关闭批次和 Hash；
- 旧直连 Step 接口不能绕过 Communication Close；
- stale、late、替换提交被拒绝；
- Close 和消息提交幂等；
- Intent 强制绑定关闭后的公司视图；
- 通信失败自动沉默；
- 隐藏消息引用被拒绝；
- Interaction Replay 的篡改检测；
- 通信失败和“服务端已接收但响应丢失”仍可完成整轮并重建审计；
- `state_only` 合法清空通信历史，Replay 按实际模型输入验证；
- 相同 Seed 下，只有收到指定私信的 B 将价格改为 12345，C/D 不改变；
- 上述私信场景稳定运行 4 Agent × 20 轮，双 Replay 均通过。

### 可重复 Acceptance

运行：

```powershell
.\.venv\Scripts\python.exe -m game_theory_agent.experiments.four_agent_acceptance `
  --provider mock `
  --rounds 20 `
  --llm-count 4 `
  --communication-mode public_private `
  --mock-communication-scenario mixed `
  --persona balanced_v1 `
  --seed 808 `
  --output runs\_interaction-p123-active-20260820
```

结果：

- 20/20 回合完成；
- 80/80 Mock 经营决策成功；
- 20 条公开提议和 20 条 A→B 私信全部进入正确视图；
- 80 条目标 Agent 消息回应全部明确记录为 `accepted`；
- 20/20 轮均满足“结构化声明 → 可见视图 → 采信 → requested action → final action”；
- 单轮只结算一次；
- 部分状态更新为 0；
- 市场 Replay 100%；
- Interaction Replay 100%；
- Acceptance `passed = true`。

结果文件：

- `runs/_interaction-p123-active-20260820/summary.json`；
- `runs/_interaction-p123-active-20260820/round-events.jsonl`。

另运行 `2 Mock + 2 Rule × 20` 私信场景（Seed 809），20/20 轮主动交互和双 Replay 同样通过，产物位于 `runs/_interaction-p123-2mock2rule-20260820/`。Rule 公司在通信指标中标记为 `not_applicable`，不再误计为主动沉默。

## 6. 真实模型 Smoke 与后续内容

已使用 `doubao-seed-2-0-lite-260215` 完成同一 Seed 810 的 `2 LLM + 2 Rule × 5轮` 对照：`off` 与 `public_private` 均通过全部协议检查和双 Replay。通信组产生 10 条公开消息、0 条私信，并记录 5 次采信和 6 次忽略。详细结果见 `docs/05-交互合作与综合实验/03-phase2-real-llm-smoke-seed810.md`。

进一步完成 Seeds 810–814 的 `off/public_only/public_private` 三条件矩阵，共 15 个 Episode。100 次真实通信生成零失败，双 Replay、消息可见性和引用检查全部通过；所有 100 条消息仍为公开 `statement`，没有真实私信或结构化对手动作请求。另完成固定状态的无消息/韧性提议/拒绝投入/Prompt Injection 四条件反事实。详细结果见 `docs/05-交互合作与综合实验/04-phase2-real-llm-5seed-smoke.md`。

以下仍需单独执行：

- 前端公开频道和私信查看界面；
- 多次独立重复与公司位置轮换后的正式通信效应实验；
- 最小 Shared Resilience Cooperation MVP；
- 多波次谈判、合作人格、约束合同和不完全市场信息。

真实 LLM 是否提高利润、增加合作或造成价格趋同，不是 P1/P2/P3 的工程验收门槛。
