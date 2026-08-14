# Agent Gateway API v1

本文档定义后期 Agent 接入 MVP 市场模型的最小协议。核心原则是：**Agent 获取信息并提交决策意图，但不能直接修改市场状态，也不能触发结算。**

## 1. 服务与信任边界

| 端口 | 服务 | 面向对象 | 能力 |
| --- | --- | --- | --- |
| `3210` | Web UI | 人类玩家 | 查看、编辑和提交玩家决策 |
| `8011` | Agent Gateway | 外部 Agent | 读取观察、读取动作契约、提交意图 |
| `8010` | Market Engine / Controller | 受信任编排器 | 创建 Episode、统一解析动作、结算与校准 |

默认全部绑定 `127.0.0.1`。这次实现没有修改 Windows 防火墙，也没有把服务暴露到公网。若以后运行在容器或局域网，建议只将 `8011` 放入 Agent 可访问的网络，`8010` 应放在私有控制平面并使用反向代理、TLS 和身份认证。

交互顺序：

```text
Agent GET observation
  -> Agent POST intent（状态不变）
  -> 受信任 Controller 读取并选择 intents
  -> 统一 decision-policy 纠正/约束每个意图
  -> MarketEnv.step 一次性结算联合动作
  -> Agent GET 下一版本 observation
```

统一规则版本当前为 `decision-policy-v1.0.0`。身份字段 `action_id`、`episode_id`、`agent_id`、`round` 和 `state_version` 由控制器生成或覆盖，Agent 只能请求经济参数。

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
- `creation_boundary`：明确 Agent Gateway 不能创建 Episode，创建只属于私有 Controller。

该接口只负责发现选项，不会修改市场状态。外部 Agent 可以据此向编排器声明实验偏好，但不能直接启动或结算市场。

### `GET /v1/episodes/{episode_id}/companies/{company_id}/observation`

返回：

- Episode、Seed、回合、`state_version` 与 `state_hash`；
- `episode_config`：该局实际采用的 `max_rounds`、固定 Seed 和市场模型；
- 公开市场指标、事件、风险预警和竞争者公开数据；
- 本公司完整状态、规则化公司分析和当前动作约束。
- `public_history`：已经结算的逐轮公开市场、本公司动作与结果、运营成本拆分、事件影响解释及到期预警结果；
- Episode 结束后的 `terminal_summary`：综合价值榜、总资产榜、估值构成，以及本公司两种终局名次。

Agent 必须将响应中的 `round` 与 `state_version` 原样带入提交意图。竞争者的现金、成本、内部投入和事故等非公开字段不在 `public_companies` 中。

终局状态保留协议字段 `round=max_rounds+1`，但同时明确返回 `decision_round=null` 和 `last_settled_round=max_rounds`，Agent 不应再尝试提交动作。风险信号到期后会在 `public_history[].resolved_signal_outcomes` 中标记为 `realized` 或 `not_realized`，不会再无解释地消失。

### 私有创建接口 `POST /api/episodes`（8010）

由受信任编排器调用，主要实验字段如下：

```json
{
  "episode_seed": 424242,
  "market_model": "quality_oriented",
  "max_rounds": 10,
  "company_ids": ["company_A", "company_B", "company_C", "company_D"]
}
```

`episode_seed` 也可以省略或传 `null` 生成随机 uint64。非法轮数、Seed 或市场值由请求 Schema 返回 `422`，不会静默纠正。

时间口径：顶层 `market` 是当前决策轮可见的条件，因此包含当前轮已激活事件对应的供应压力；`public_history[].market` 是某个已结算回合的结果，只使用该回合开始时已激活的事件。两者不得混用。控制平面的 step 响应另外返回 `settled_market`，供前端记录同轮历史。

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
    "incident_response": {
      "mode": "wait",
      "repair_budget_cents": 0
    },
    "strategy_summary": "保持价格，针对下一轮预警配置韧性"
  },
  "rationale": "当前份额稳定，风险预警概率较高。",
  "expected_outcome": "控制事件损失并保留现金。"
}
```

响应中的 `resolution` 是统一规则生成的预览。`adjustments` 会说明边界修正、品牌/服务饱和、低利用率停止扩产、末轮禁用长期投资或现金缩放。最终执行时 Controller 会基于同一个状态版本再次解析。

过期的 `round` 或 `state_version` 返回 `409 STALE_OBSERVATION`，Agent 应重新读取 observation 后再规划。

### `GET /v1/episodes/{episode_id}/intents/{intent_id}`

返回意图的 `accepted`、`executed` 或 `rejected` 状态及最终解析结果。

## 3. 受保护的 Controller 接口

以下接口**不在 8011 上挂载**。调用必须位于 8010，并携带：

```http
X-Controller-Token: <MARKET_CONTROLLER_TOKEN>
```

如果服务端没有设置 `MARKET_CONTROLLER_TOKEN`，Controller 接口返回 `503`；令牌错误返回 `401`。

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

`fallback=rule` 表示没有 Agent 意图的公司由透明规则策略补全；`fallback=error` 表示缺少任何公司意图都停止结算。响应包含每家公司的 `decision_resolutions` 与实际执行的 `executed_intent_ids`。

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
