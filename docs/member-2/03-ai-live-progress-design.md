# AI 调用与生成过程实时信息设计

日期：2026-08-23

## 目标

在真实回合运行期间，让参与者、观察者和研究员看到可核验的 PersonaAgent/SABM 执行进度，而不是只有统一的“处理中”。所有信息必须来自现有 `progress_callback` 节点事件或后端计时，不得用前端定时器推测节点、生成候选、Token 或成功状态。

## 非目标与安全边界

- 不流式展示模型生成的原始文本、未完成 JSON、隐藏推理或 Provider 原始响应。
- 不返回 API Key、Controller 凭据、请求头或其他敏感字段。
- 不用预计 Token、虚构百分比或固定阶段文案冒充已发生事件。
- 不改变 Controller 结算、Intent 校验和规则 fallback 语义。
- 不增加 WebSocket、SSE 或新的外部依赖。

## 方案选择

采用短轮询进度接口。现有 `POST /api/episodes/{episode_id}/managed-rounds` 保持同步和向后兼容；前端在等待该请求时，每 500 毫秒读取一次只读进度快照。相比 SSE，该方案可以直接复用当前 FastAPI 与线程池执行结构，页面重载后也能恢复当前状态；相比 Provider Token 流，它不会暴露未校验内容。

## 后端运行状态

后端维护进程内、按 Episode 隔离的最新管理回合状态。状态生命周期为：

1. `idle`：本 Episode 尚未运行管理回合；
2. `running`：已取得回合锁，模型公司处于 queued/running/completed/failed；
3. `settling`：所有模型任务结束，正在提交人类 Intent 或调用 Controller；
4. `completed`：Controller 已完成结算；
5. `failed`：回合在结算前异常退出，保存安全错误类别。

每次新回合覆盖上一次“最新进度”，但最终 Episode history 仍由现有 append-only transition 保存。状态写入与读取使用专用锁；并发模型线程只更新自己公司的记录，不能覆盖其他公司事件。

### 公司进度字段

每家公司保存：

- `company_id`、安全模型显示名；
- `status`：`queued | running | completed | fallback | failed`；
- `current_stage`：最近收到的真实节点事件；
- `events`：按发生顺序保存的安全 `{stage, details, occurred_at_ms}`；
- `started_at_ms`、`updated_at_ms`、`elapsed_ms`；
- `provider_attempts`：由真实 `provider_request` 事件计数；
- `provider_waiting`：最后事件是 `provider_request` 且尚无对应 response/error；
- Provider 返回后才出现的 `total_tokens`、`provider_latency_ms`、`finish_reason`；
- 最终 `fallback_used` 与安全 `error_category`。

事件详情复用当前节点层已经生成的白名单字段：attempt、repair、status、finish reason、usage availability、total tokens、latency、repair attempts 和终态。进度存储不得保存 prompt、候选正文、原始响应或凭据。

## API 契约

新增：

```http
GET /api/episodes/{episode_id}/managed-rounds/progress
```

响应示例：

```json
{
  "episode_id": "episode-1",
  "round": 1,
  "state_version": 0,
  "status": "running",
  "started_at_ms": 1787460000000,
  "updated_at_ms": 1787460003500,
  "elapsed_ms": 3500,
  "companies": {
    "company_B": {
      "company_id": "company_B",
      "status": "running",
      "current_stage": "provider_request",
      "events": [
        {"stage": "load_snapshot", "details": {}, "occurred_at_ms": 1787460000100},
        {"stage": "provider_request", "details": {"attempt": 1, "repair": false}, "occurred_at_ms": 1787460000400}
      ],
      "elapsed_ms": 3100,
      "provider_attempts": 1,
      "provider_waiting": true,
      "total_tokens": null,
      "provider_latency_ms": null,
      "fallback_used": null,
      "error_category": null
    }
  }
}
```

Episode 不存在时返回 404。尚未运行时返回 `idle` 和空 `companies`，而不是 404。进度接口只读，不需要浏览器持有 Controller Token。`POST managed-rounds` 成功或失败时必须在 `finally` 路径写入终态，避免页面永久停留在 running。

## 前端数据流

提交真实回合时：

1. 清空上一轮临时进度；
2. 发起原 `POST managed-rounds`；
3. 同时每 500 毫秒轮询进度接口；
4. 收到运行快照后更新独立 `liveProgress`，不修改权威市场状态；
5. POST 完成后停止轮询，以最终 `executions` 和 Controller payload 更新页面；
6. POST 失败或组件卸载时停止轮询，并显示后端实际错误。

轮询网络错误不会中断正在运行的 POST；界面保留最后一份真实快照并标记“进度暂时不可用”。不得因一次轮询失败回退到演示数据。

## 页面展示

### 01 实时现场

紧凑进度卡为每个 AI 显示：公司、当前节点中文名、状态、已用时间和模型调用次数。当 `provider_waiting=true` 时明确显示“模型已接收请求，等待结构化结果”。Provider 返回后显示实际 Token 和调用耗时；校验、修复、Intent 提交和 fallback 均使用真实事件更新。

### 02 智能体观察

三列/四列节点流使用实时事件：最近进入且尚未被下一节点替代的节点显示“执行中”，更早节点显示“已执行”，未出现节点显示“尚未运行”，错误事件对应节点显示“错误”。节点展开区在运行中只显示安全事件详情；回合结束后继续使用完整终态 trace 显示 prompt audit、候选和 Intent 回执。

演示模式保持独立，不调用进度接口。三个真实入口共享同一个轮询和展示实现，差异仅为参与者有无人工动作。

## 错误与恢复

- Provider 错误：显示安全 error category、实际耗时和是否进入修复或 fallback，不显示原始错误正文。
- Controller 结算错误：全局状态为 failed，已完成的公司事件仍可查看。
- 重复提交：现有 409 保持不变；进度接口继续返回正在执行的回合。
- 页面切换：轮询挂在真实回合任务而非具体导航页，切换 `01/02` 不丢失进度。
- 页面刷新：可以读取后端当前最新进度；前端不会恢复已丢失的 POST Promise，因此只读展示到终态，并要求用户从 Episode 状态继续。

## 测试与验收

- 后端单元测试覆盖 idle、逐事件更新、并发公司隔离、provider response/error、completed、failed 和安全字段白名单。
- API 测试用阻塞模型模拟真实等待，在 POST 未完成时断言 GET 能看到 `provider_request` 与 `provider_waiting=true`，完成后断言 Token、耗时和终态。
- 前端测试覆盖轮询启停、临时网络失败、节点 current/done/error 映射，以及 POST 完成后终态 trace 替换实时快照。
- 在用户可见浏览器分别运行参与者、观察者和研究员真实回合，确认模型等待阶段能看到真实调用信息，三列/四列节点按事件推进，最终数据与 Controller 响应一致。
- 全量执行后端 pytest、前端 lint/test/build，并检查常规 UI 除允许的 AI 错误外不新增 Provider 专用字段。
