# Agent Gateway API v1

本文档定义后期 Agent 接入 MVP 市场模型的最小协议。核心原则是：**Agent 获取信息并提交决策意图，但不能直接修改市场状态，也不能触发结算。**

## 1. 服务与信任边界

| 端口 | 服务 | 面向对象 | 能力 |
| --- | --- | --- | --- |
| `3210` | Web UI | 人类玩家 | 查看、编辑和提交玩家决策 |
| `8011` | Agent Gateway | 外部 Agent | 读取公司观察、提交非约束消息、读取关闭视图、提交意图 |
| `8010` | Market Engine / Controller | 受信任编排器 | 创建 Episode、统一解析动作、结算与校准 |

默认全部绑定 `127.0.0.1`。这次实现没有修改 Windows 防火墙，也没有把服务暴露到公网。若以后运行在容器或局域网，建议只将 `8011` 放入 Agent 可访问的网络，`8010` 应放在私有控制平面并使用反向代理、TLS 和身份认证。

交互顺序：

```text
Agent GET frozen observation
  -> Agent POST communication submission（状态不变）
  -> Controller Communication Close
  -> Agent GET observation with company-scoped closed view
  -> Agent POST intent with observation_hash + communication_view_digest（状态不变）
  -> 受信任 Controller 读取并选择 intents
  -> 统一 decision-policy 纠正/约束每个意图
  -> MarketEnv.step 一次性结算联合动作
  -> Agent GET 下一版本 observation
```

统一规则版本当前为 `decision-policy-v1.1.0`。身份字段 `action_id`、`episode_id`、`agent_id`、`round` 和 `state_version` 由控制器生成或覆盖，Agent 只能请求经济参数。

## 2. 在线接口说明

- Agent Swagger UI：`http://127.0.0.1:8011/docs`
- Agent OpenAPI JSON：`http://127.0.0.1:8011/openapi.json`
- 控制平面 Swagger UI：`http://127.0.0.1:8010/docs`
- 控制平面 OpenAPI JSON：`http://127.0.0.1:8010/openapi.json`

### `GET /health`

Agent Gateway 健康检查。`execution_access` 永远为 `false`。

### `GET /v1/capabilities`

返回协议版本、Agent 可用能力、控制器拥有的字段以及决策策略版本。Agent 应在启动时读取一次。

### `GET /v1/episode-options`

无需 Episode 即可读取受支持的实验配置：

- `round_options`：当前为 `5 / 10 / 15 / 20`；
- `seed`：`0`～`2^64-1`，创建请求省略或传 `null` 表示由 Controller 随机生成，传整数表示固定 Seed；
- `market_models`：`random / balanced / value_oriented / quality_oriented / service_oriented` 及公开说明；
- `information_modes`：`perfect` 是完全信息基线；`public` 使用 `visibility-public-v2.0.0`，只公开市场结果和对手价格、份额、销量、声誉；
- `creation_boundary`：明确 Agent Gateway 不能创建 Episode，创建只属于私有 Controller。

该接口只负责发现选项，不会修改市场状态。外部 Agent 可以据此向编排器声明实验偏好，但不能直接启动或结算市场。

### `GET /v1/episodes/{episode_id}/companies/{company_id}/observation`

返回：

- Episode、回合、`state_version` 与 `state_hash`；随机 Seed 不向 Agent 暴露；
- `public_state`、仅属于本公司的 `private_state`、`visibility_policy_version` 和完整字段矩阵；
- `observation_hash`：覆盖整份实际 Observation、Visibility Policy 版本和 Belief 版本；
- `episode_config`：该局实际采用的 `max_rounds`、市场模型和信息模式；
- 公开市场指标、事件、风险预警和竞争者公开数据；
- `market_regime`：确定性代码生成的竞争、需求、产能、成本、集中度和风险状态摘要；
- `decision_support`：确定性代码生成的单位贡献、保本价格、固定成本、现金跑道、安全预算与战略阶段；
- 本公司完整状态、规则化公司分析和当前动作约束。
- `communication_mode` 与 `communication_view`：通信关闭前视图为 `null`；关闭后只返回该公司有权看到的公开消息、私信和 `view_digest`；
- `communication_history`：最近三轮由权威关闭批次派生、按同一公司权限过滤的历史视图；
- `cooperation_mode`、`shared_resilience` 与 `cooperation`：启用 `shared_resilience_v1` 时，分别提供公共韧性状态，以及本公司可见的提议、承诺、履约历史和公共可信度；
- `belief_schema_version`、`belief_hash` 与 `belief_state`：`public_action_v1` 只根据已结算公开价格给出对手本轮降价/持平/涨价概率；`public_action_signal_v2` 额外融合本公司合法可见的结构化非绑定价格声明，并按历史言行一致率降权；
- `opponent_model_state / opponent_model_hash`：仅在 `opponent_model_mode=public_strategy_v1` 时出现，只使用公开价格、销量、份额、声誉与公开韧性贡献；
- `utility_inference_state / utility_inference_hash`：仅在 `utility_inference_mode=strategy_utility_v1` 时出现，并绑定 Opponent Model Hash；
- `game_theory_advice`：在 `advisor_mode=bayesian_price_v1` 或 `bayesian_strategy_v2` 时出现，是带 Hash/Replay 的非绑定近似建议，不是 Final Action；v2 明确不是 Nash 求解器；
- `repeated_game_strategy / repeated_game_strategy_hash`：仅在 `repeated_game_mode=reciprocity_v1` 且合作开启时出现，从权威合作记忆派生，不直接执行贡献；
- `competitors`：`perfect` 模式返回对手完整 CompanyState，`public` 模式仅返回对手公开摘要；
- `public_history`：已经结算的逐轮公开市场、本公司动作与结果、运营成本拆分、事件影响解释及到期预警结果；
- Episode 结束后的 `terminal_summary`：综合价值榜、总资产榜、估值构成，以及本公司两种终局名次。

Agent 必须将响应中的 `round`、`state_version` 和 `observation_hash` 原样带入提交意图。通信开启时还必须回传关闭后的 `communication_view_digest`。任一视图绑定不一致都会返回 409，Intent 不会进入结算。`public_companies` 在所有模式下都只包含公开摘要；完整对手信息只会在 `perfect` 的 `competitors` 出现。不要从 `public_companies` 推断当前信息模式，应读取顶层 `information_mode`。

研究型不对称信息实验可以在市场级 `information_mode=public` 时，由受信任 Controller 额外设置 `observer_information_modes={"company_A": "perfect"}`。此时只有 A 的顶层 `information_mode` 和 `competitors` 按完全信息策略生成，其他公司仍为 public；公共状态不变。该处理会写入 Manifest 并由公司级 Information Replay 验证。由于它暴露对手完整私有状态，创建和读取都强制使用 Controller/Agent Token，不是普通前端开关。

当前 `public` 摘要不包含对手收入、服务质量、财务、运营、私人韧性、事故、Persona 或计划；market 也不暴露价格锚点、需求偏置或消费者效用权重。`belief_mode=off` 时 `belief_schema_version=none`；v1/v2 Belief、Opponent/Utility 推断和两代 Advisor 都不能解释为精确 Bayesian Nash 求解器。v2 消息 Signal 固定 `verified_fact=false`，不能写回 Public/Private State。

`episode_seed` 只保存在私有 Controller、EpisodeManifest、MarketTransition 和 Replay 数据中。Agent 只能通过风险信号观察不确定性，不能读取 Seed 或未来随机抽样结果。

终局状态保留协议字段 `round=max_rounds+1`，但同时明确返回 `decision_round=null` 和 `last_settled_round=max_rounds`，Agent 不应再尝试提交动作。风险信号到期后会在 `public_history[].resolved_signal_outcomes` 中标记为 `realized` 或 `not_realized`，不会再无解释地消失。

### 私有创建接口 `POST /api/episodes`（8010）

由受信任编排器调用，主要实验字段如下：

```json
{
  "episode_seed": 424242,
  "market_model": "quality_oriented",
  "max_rounds": 10,
  "information_mode": "perfect",
  "observer_information_modes": {},
  "communication_mode": "public_private",
  "cooperation_mode": "shared_resilience_v1",
  "belief_mode": "public_action_v1",
  "company_ids": ["company_A", "company_B", "company_C", "company_D"],
  "agent_configs": {
    "company_A": {
      "model_name": "provider/model-version",
      "persona": {
        "persona_id": "selfish_long_term",
        "profile_hash": "sha256:..."
      }
    }
  }
}
```

通信、合作或 Belief treatment 任一开启时，创建请求必须携带 `X-Controller-Token`。创建响应只返回一次 company-scoped Agent token；Belief Observation 同样要求对应公司的 token：

```json
{
  "agent_tokens": {
    "company_A": "one-time-token-A",
    "company_B": "one-time-token-B"
  },
  "agent_token_header": "X-Agent-Token"
}
```

Session 只保存 Token 的 SHA-256。通信开启时，公司 Observation、通信提交、通信视图和 Intent 都必须使用与该公司绑定的 `X-Agent-Token`。

`shared_resilience_v1` 支持 `communication_mode=off`（无消息贡献基线）或 `public_private`（完整提议链路）；不支持 `public_only`。即使通信关闭，只要合作动作开启，公司 Observation 和 Intent 仍需要公司 Token，以保证贡献归属不可伪造。

`episode_seed` 也可以省略或传 `null` 生成随机 uint64。非法轮数、Seed 或市场值由请求 Schema 返回 `422`，不会静默纠正。

`agent_configs` 是只进入 EpisodeManifest 的实验审计元数据，不进入 MarketState 或市场公式。若为公司配置了 `agent_id`，Intent 的该字段必须匹配；未配置的旧 Episode 由服务端使用固定 company ID 作为审计身份，客户端自报值不会成为权威身份。人格的完整 Runtime 配置和逐轮效用协议见 `docs/04-Persona研究/01-persona-research.md`。

时间口径：顶层 `market` 是当前决策轮可见的条件，因此包含当前轮已激活事件对应的供应压力；`public_history[].market` 是某个已结算回合的结果，只使用该回合开始时已激活的事件。两者不得混用。控制平面的 step 响应另外返回 `settled_market`，供前端记录同轮历史。

### `POST /v1/episodes/{episode_id}/companies/{company_id}/communication/submissions`

提交本公司在当前同步波次的非执行性消息草稿。请求必须携带公司专属 `X-Agent-Token`：

```json
{
  "round": 3,
  "state_version": 2,
  "state_hash": "sha256:...",
  "submission": {
    "schema_version": "communication-submission-v1.0.0",
    "messages": [
      {
        "channel": "private",
        "recipients": ["company_B"],
        "speech_act": "proposal",
        "content": "建议本轮保持价格。",
        "requested_peer_action": {"price_cents": 10500}
      }
    ]
  }
}
```

每家公司每轮最多一条公开消息和一条一对一私信。消息绑定当前 `round/state_version/state_hash`，不改变市场状态；相同提交幂等，不同替换提交返回 `409`，Close 后提交返回 `409`。

合作提议只能放在 private `proposal` 消息的 `cooperation_proposal` 中；回应只能放在后续轮次 private `response` 消息的 `cooperation_response` 中。v1 只支持 `shared_resilience`，同一同步波次不能立即回应本轮新提议。

### `GET /v1/episodes/{episode_id}/companies/{company_id}/communication/view`

Communication Close 后返回该公司的关闭视图。公开消息对所有公司可见；私信只对发送者和唯一收件人可见。关闭前返回 `409 COMMUNICATION_NOT_CLOSED`。请求必须携带公司专属 Token，其他公司的 Token 不能读取该视图。

### `GET /v1/episodes/{episode_id}/companies/{company_id}/action-contract`

返回当前回合允许提交的字段、数值边界、可用现金、事故维修上限，以及哪些字段属于 Controller。

### `POST /v1/episodes/{episode_id}/intents`

提交非执行性意图。成功返回 `202 Accepted`；响应中的 `executed` 为 `false`，市场状态与 `state_hash` 不变。请求示例：

```json
{
  "agent_id": "planner-agent-01",
  "company_id": "company_A",
  "round": 3,
  "state_version": 2,
  "requested_action": {
    "price_cents": 10100,
    "advertising_budget_cents": 600000,
    "service_budget_cents": 700000,
    "capacity_investment_cents": 0,
    "resilience_budget_cents": 300000,
    "shared_resilience_contribution_cents": 1000000,
    "incident_response": {
      "mode": "wait",
      "repair_budget_cents": 0
    },
    "strategy_summary": "保持价格，针对下一轮预警配置韧性"
  },
  "rationale": "当前份额稳定，风险预警概率较高。",
  "expected_outcome": "控制事件损失并保留现金。",
  "communication_view_digest": "sha256:..."
}
```

响应中的 `resolution` 是统一规则生成的预览。`adjustments` 会说明边界修正、品牌/服务饱和、低利用率停止扩产、末轮禁用长期投资或现金缩放。最终执行时 Controller 会基于同一个状态版本再次解析。

过期的 `round` 或 `state_version` 返回 `409 STALE_OBSERVATION`，Agent 应重新读取 observation 后再规划。通信开启时，Communication 尚未关闭或 `communication_view_digest` 不匹配也会返回 `409`，因此 Intent 不能在关闭前进入结算。

`shared_resilience_contribution_cents` 是唯一会真实扣款并形成公共韧性的合作动作。Proposal、Response 和 Commitment 本身不会修改市场。

### `GET /v1/episodes/{episode_id}/intents/{intent_id}`

返回意图的 `accepted`、`executed` 或 `rejected` 状态及最终解析结果。

## 3. 受保护的 Controller 接口

以下接口**不在 8011 上挂载**。调用必须位于 8010，并携带：

```http
X-Controller-Token: <MARKET_CONTROLLER_TOKEN>
```

如果服务端没有设置 `MARKET_CONTROLLER_TOKEN`，Controller 接口返回 `503`；令牌错误返回 `401`。

### `POST /api/v1/controller/episodes/{episode_id}/communication/close`

由编排器原子关闭当前通信阶段：

```json
{
  "round": 3,
  "state_version": 2,
  "state_hash": "sha256:..."
}
```

响应包含完整 Controller 审计 Closure、`transcript_hash`、各公司 `view_digest` 和可见消息 ID。相同状态的重复 Close 幂等；关闭本身不修改 `MarketState` 或 `state_hash`。

当 `cooperation_mode=shared_resilience_v1` 时，响应还包含 `cooperation_close`，其中记录本轮生成的权威 Proposal、Response 与非约束 Commitment。结算响应随后包含 `cooperation_round`，记录实际贡献、公共韧性更新、履约/背离与可信度更新。

### `POST /api/v1/controller/episodes/{episode_id}/settle-agent-round`

由编排器为每家公司选择一个已接收意图，再统一结算：

```json
{
  "step_id": "episode-123:3:2",
  "intent_ids": {
    "company_A": "intent-...",
    "company_B": "intent-..."
  },
  "fallback": "rule"
}
```

`fallback=rule` 表示没有 Agent 意图的公司由透明规则策略补全；`fallback=error` 表示缺少任何公司意图都停止结算。响应包含每家公司的 `decision_resolutions` 与实际执行的 `executed_intent_ids`。相同 `step_id` 与相同请求可安全重试并返回首次结果；相同 `step_id` 搭配不同意图会返回 `409`。

通信开启时，旧 `/api/episodes/{episode_id}/steps` 和 `/player-steps` 直连路由返回 `409 INTERACTION_REQUIRES_AGENT_BARRIER`，不能绕过消息关闭、视图绑定和受保护结算。

### `POST /api/v1/controller/evaluations/presets`

运行多 Seed 档位校准：

```json
{ "seed_start": 0, "seed_count": 200 }
```

输出每个投入档位的排名分布、第一/第四比例和 70% 极端排名阈值是否通过。该接口计算量较高，不应开放给普通 Agent。

## 4. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MARKET_API_HOST` | `127.0.0.1` | 私有市场引擎绑定地址 |
| `MARKET_API_PORT` | `8010` | 私有市场引擎端口 |
| `MARKET_AGENT_GATEWAY_ENABLED` | `1` | `0` 时不启动 Agent Gateway |
| `MARKET_AGENT_HOST` | `127.0.0.1` | Agent Gateway 绑定地址 |
| `MARKET_AGENT_PORT` | `8011` | Agent Gateway 端口 |
| `MARKET_CONTROLLER_TOKEN` | 无 | 启用受保护结算/校准接口所需的高熵令牌 |

PowerShell 本地启动示例：

```powershell
$env:PYTHONPATH="src"
$env:MARKET_CONTROLLER_TOKEN="请替换为随机高熵令牌"
python -m game_theory_agent.api
```

## 5. 稳定性与安全约定

- Agent 不应缓存 observation 跨 `state_version` 使用。
- 同一轮可以提交多个候选意图，但只有 Controller 显式选择的意图会执行。
- Agent Gateway 当前使用进程内存储，进程重启后 Episode 和意图都会丢失；生产化前应增加持久化队列和幂等键。
- 本地 MVP 的 `agent_id` 是声明值，不等价于可靠身份。跨主机开放前必须在网关前增加认证，并把认证主体映射到允许控制的 `company_id`。
- 不要将 `8010` 暴露到 Agent 网络；Token 是第二道防线，不替代网络隔离。
