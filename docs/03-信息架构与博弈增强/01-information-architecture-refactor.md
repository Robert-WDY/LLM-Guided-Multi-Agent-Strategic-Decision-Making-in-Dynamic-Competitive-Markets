# Information Architecture Refactor（Stage 4.5 / Phase A）

## 阶段目标

本阶段只建立进入不完全信息研究之前的可信信息基础设施：

```text
True MarketState
  -> versioned VisibilityPolicy
  -> company-scoped Observation
  -> Observation Hash
  -> Agent Context / Intent binding
  -> Information Replay
```

Phase A 当时不实现 Belief Ledger、Opponent Model 或 Bayesian Best Response。默认 Observation 仍以 `belief_schema_version=none` 和 `belief_state=null` 表示无信念基线；后续 Phase B 的可选公开动作信念见 `docs/03-信息架构与博弈增强/02-belief-mvp.md`。

## 单一事实来源

`MarketEnv` 只负责真实状态和 `Action -> Next State`。原 `MarketEnv.get_observation()` 已移除。所有 `TrueState -> Agent View` 都由 `agents/observation.py::ObservationBuilder` 完成；API 和离线 Persona 实验使用同一个 Builder。

MarketState 没有被拆成三份。PublicState 和 PrivateState 是不可执行的视图契约：

- `public_state`：所有公司在同一状态版本下完全一致；
- `private_state`：只包含当前公司自己的完整 CompanyState；
- Controller-only：Episode Seed、未来随机抽样、对手财务/运营/事故、对手 Persona 和计划不进入 public 视图。

## Visibility Policy

当前有两个版本化处理：

| 模式 | Policy | 对手信息 |
|---|---|---|
| `perfect` | `visibility-perfect-v1.0.0` | 完整 CompanyState，作为完全信息基线 |
| `public` | `visibility-public-v2.0.0` | 仅公司 ID、价格、份额、销量和声誉 |

`public` 不再暴露：

- `round_revenue_cents`；
- `service_quality_ppm`；
- 财务、成本、现金和利润；
- 产能、利用率和内部运营能力；
- 私人韧性和公司事故；
- Persona、历史完整动作和计划；
- 市场需求偏置、价格锚点、价格带和消费者效用权重；
- 市场事件的精确传导倍率。

Public market 保留已经实现或公开宣布的需求、Outside Option、缺货、市场情绪、实际供应成本指数、平均成交价和市场模型说明。公共韧性、风险预警、粗粒度已激活事件及公开贡献继续可见。

## Observation 与 Hash

Phase A API Observation 升级为 `agent-observation-v1.7.0`，新增：

- `visibility_policy_version`；
- 完整 `visibility_policy`；
- `public_state`；
- `private_state`；
- `belief_schema_version` / `belief_state`；
- `observation_hash`。

Hash 协议为 `observation-view-hash-v1.0.0`：

```text
SHA256(
  hash_protocol_version
  + visibility_policy_version
  + belief_schema_version
  + canonical observation without observation_hash
)
```

Hash 是一致性校验，不是数字签名。拥有日志写权限的人若同时修改内容和 Hash，普通 Hash 校验本身无法证明真实性；Information Replay 还会使用权威 True State 和 Manifest 重新执行 Visibility Policy，因此“重新计算一个伪造 Hash”仍不能通过 Replay。

## Agent 与 Controller 绑定

`DecisionMeta` 携带 `observation_hash`。模型生成 Intent 后必须原样回传：

```json
{
  "round": 3,
  "state_version": 2,
  "observation_hash": "sha256:...",
  "communication_view_digest": "sha256:...",
  "requested_action": {}
}
```

Controller 在接收 Intent 时重新生成当前公司的关闭后 Observation。Hash 不一致返回 `409 OBSERVATION_VIEW_MISMATCH`。这与 `communication_view_digest` 一起绑定“哪个公司、哪个状态、哪份消息视图”产生了该意图。

Coordinator 在调用模型前执行 Information 校验，而不是在市场结算后才发现错误。通信生成和最终经营决策分别保存各自的 `ObservationSnapshot`，因为 Communication Close 会改变消息视图及完整 Observation Hash，但不会改变 MarketState Hash。

## Information Replay

`verify_information_replay(events, manifest)` 对每个模型输入重建：

1. 验证 True State 自身 Hash；
2. 验证 Episode/round/state version/state hash/company 绑定；
3. 按 Manifest 指定的 information mode 和 policy 重建公司视图；
4. 精确比较 PublicState、PrivateState、competitors、market、events 等可见性字段；
5. 验证所有公司 PublicState 一致；
6. 验证 Observation Hash；
7. 验证模型实际 Decision/Communication Context 与该 Observation 相同；
8. 拒绝 public history 或 market regime 中绕过 Policy 的私有字段。

`agent-round-event-v1.7.0` 的非禁用模型输入必须包含 Snapshot。v1.0–v1.6 日志仍可解析，但不会被错误宣称为拥有新 Information Replay 证据。

## 验收结果

确定性测试覆盖：

- public 对手财务、运营、Persona、服务能力与内部市场参数泄漏为 0；
- 同一 True State 下所有公司的 PublicState 完全相同；
- 每家公司 PrivateState 只属于自己；
- perfect/public 投影不改变 MarketState；
- 内容篡改且不改 Hash 会失败；
- 注入对手现金并重新计算 Hash 仍会被 TrueState 重建拒绝；
- 缺 Snapshot 的 v1.7 RoundEvent 会失败；
- 错误 Intent Observation Hash 返回 409；
- Communication 和 Decision 两类模型输入均可重建。

运行产物：

- `runs/information-refactor-p0-acceptance-20260821-v1/summary.json`；
- `runs/information-refactor-p0-acceptance-20260821-v1/round-events.jsonl`；
- `runs/cooperation-research-v2-p0-acceptance-20260821-information-v1/summary.json`。

4 Mock × 5 轮 public + public_private 验收通过 Economic、Interaction 和 Information Replay；共重建 40 份模型输入（20 次通信生成 + 20 次经营决策），无 fallback。Cooperation 验收同时通过 Economic、Interaction、Information 和 Cooperation 四层 Replay。

## 下一阶段边界

Phase B 已在后续 `agent-observation-v1.8.0` 中实现：

- company-scoped Belief Ledger；
- 确定性 Evidence；
- 行为频率或 Bayesian 规则更新；
- price cut / maintain / raise 概率；
- Belief Hash 和 Belief Replay；
- Brier Score、Log Loss 和 Calibration Error。

实现结果、算法、Replay、校准指标和边界见 `docs/03-信息架构与博弈增强/02-belief-mvp.md`。即使 Phase B 已完成，`public_action_v1` 仍只是简单公开行为频率信念，不能称为完整 Bayesian Agent。
