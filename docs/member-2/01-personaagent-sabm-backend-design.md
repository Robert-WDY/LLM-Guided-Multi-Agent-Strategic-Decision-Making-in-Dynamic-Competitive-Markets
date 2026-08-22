# PersonaAgent / SABM 后端设计

## 目标与非目标

本实现把 `~member-2-v1` 固定提交 `7cbace1` 中的 OpenRouter 决策能力，以最小范围接入源项目后端。实现只修改 Provider 和 AI 决策路径，并增加 Windows 启动入口与后端文档。

明确不包含：

- 前端 Agent Lab、Research Lab 或其他 UI 替换；
- 新的单/多 Agent Lab、Job、Research Lab 公共 API；
- `agents/progress`、`agents/research` 或扩展 orchestration；
- 对市场状态、Controller 结算规则和 Agent Gateway 权限边界的修改。

## 组件职责

| 组件 | 职责 |
|---|---|
| `OpenRouterProvider` | 校验模型 allowlist、构造请求、解析结构化结果、隐藏凭据和原始错误正文 |
| `PersonaAgent` | 把公司身份、Persona Profile、模型和单公司 SABM runtime 绑定为一个后端 Agent |
| `SingleAgentRuntime` | 使用 company/episode 隔离的 thread ID 执行有界 LangGraph；每轮显式重置临时状态 |
| `SABMEpisodeRunner` | 并发运行多家公司节点流，汇总已接受 intent，并调用原 Controller 结算 |
| `AgentGatewayClient` | 读取公司观察/动作契约并提交 intent，不获得 Controller 权限 |
| `JsonlTraceStore` / SQLite checkpointer | 记录审计 trace 和节点 checkpoint，显式允许本地 Agent 模型类型反序列化，默认位于非 Git 中间产物目录 |

`PersonaAgent` 使用源项目 `PersonaProfile`，把 label、objective、效用权重和 traits 写入受长度约束的 system prompt；市场事实仍只来自 Agent Gateway，Persona 不得覆盖动作约束或身份字段。
版本化 Persona manifest 与 profile hash 同步进入每轮 trace，便于把候选、动作与具体实验人格对应起来。

## SABM 节点流

单轮固定执行以下节点：

1. `load_snapshot`：同时读取 observation 和 action contract，并校验 episode/company/round/state_version 一致。
2. `build_context`：构造有界观察、最近两轮公开历史和已保存 trace。
3. `reflect_strategy`：从可见历史生成确定性策略反思，不增加模型调用。
4. `generate_candidates`：OpenRouter 生成恰好三个结构化候选方案。
5. `validate`：校验现金、动作边界、证据路径和候选结构。
6. `repair_decision`：首次校验失败时最多进行一次模型修复。
7. `prepare_intent`：把选中候选绑定到当前 snapshot key。
8. `submit_intent`：经 Agent Gateway 提交，不直接调用 Controller。
9. `finalize`：写入状态、Persona manifest、候选、错误码、token/latency 和 receipt trace。

失败、无效输出、过期 observation 或提交结果未知时，节点流返回非 accepted 状态，不伪造成功 intent。`SABMEpisodeRunner` 只汇总 `status=accepted` 且存在 `intent_id` 的结果；其余公司不传 intent，由 Controller 的 `fallback=rule` 使用原规则动作。

## 后端对接

`run_agents` 的 OpenRouter 分支执行：

1. 使用原 Controller API 创建 Episode，并把每个 `single-agent-<company_id>` 写入 manifest。
2. 为每家公司创建独立 runtime、checkpoint namespace 和 trace 目录。
3. 把 Controller 返回的一次性 Agent token 仅安装到对应 Gateway client。
4. 并发运行 PersonaAgent，按轮汇总 intent。
5. 调用原 `settle-agent-round` Controller 接口，缺失 intent 使用规则回退。

Doubao、DeepSeek、Mock 继续走源项目原 `AgentRuntime + RoundCoordinator` 路径，因此此次接入没有修改公共 API 或旧 orchestration。

`SABMEpisodeRunner` 在每轮开始读取 Controller 权威的 `round/state_version`，结算 step ID 固定为 `<episode_id>:<round>:<state_version>`。同一 Agent 的 LangGraph thread 可复用 checkpoint，但 `intent_receipt`、候选、错误和 Provider 统计等轮内字段在调用前显式清空，避免上一轮状态进入本轮 trace。

## 只读前端边界

`02 智能体观察` 展示八节点纵向拓扑，对应 `load_snapshot` 至 `finalize`；`01 实时现场` 只显示紧凑执行状态。该拓扑是独立的后端执行视图，参与者模式的三家 AI 公司按三列并排，每列内部纵向展示八节点，不复用六步处理说明的公司选择状态；因此人类公司的“接收信息”不会被呈现为 AI 节点事件。基础真实 Episode 通过 `POST /api/episodes/{episode_id}/managed-rounds` 触发后端节点流；接口并发运行所有模型公司、接收结构化人类动作、汇总 Intent 并调用 Controller 联合结算。响应中的 `executions` 包含实际 progress event，以及节点审计所需的结构化 trace。未运行时所有节点显示“尚未运行”。

真实模式的前端边界集中在 `frontend/app/real-runtime.ts`。它只消费 Episode `state`、每公司 `observations`、`decision_resolutions` 和本次会话保存的 managed-round 响应，不得导入 `DEMO_*`。模型运行前，消息、信念、计划、决策、动作、结算和历史均为空；模型返回或 Controller 结算后才分别写入安全 trace 和权威结果。参与者、观察者与研究员共用该投影，因此三种入口不会各自复制或补造数据。

管理回合执行期间，节点层现有 `progress_callback` 同时写入按 Episode/公司隔离的内存进度注册表。只读 progress API 仅保留 stage、attempt、repair、finish reason、usage、latency 和安全 error category；前端短轮询只影响过程展示，不修改权威市场状态。模型结束后，完整 execution trace 和 Controller payload 继续作为终态来源。

正常 UI 使用“AI 模型”等产品语义，不出现 Provider 名称。只有固定密钥缺失或 AI Provider 调用失败时，页面内错误可以显示 `OpenRouter` 和 `open_router-api_key.env` 以便排障。节点详情展示真实 `memory_view`、安全审计用的完整 `prompt_audit`、确定性策略反思、结构化三候选、选中理由、耗时、Token、校验/修复、`prepared_intent`、`intent_receipt`、错误码和 fallback；不得展示密钥、Provider 原始响应或隐藏推理。

## Provider 与 secrets

创建 OpenRouter API Key、写入固定本地文件及安全更换步骤见 [OpenRouter API Key 创建与本地配置](05-openrouter-api-key-setup.md)。

- 默认 secret 路径：`secrets/open_router-api_key.env`。
- 可用 `--openrouter-secret` 指定另一个本地文件。
- 文件可包含原始 key，Provider 读取时会去除空白；异常、日志和 trace 不输出 key。
- `.gitattributes` 对 `secrets/**` 强制 `filter=git-crypt diff=git-crypt -text`；仅以 `.example` 结尾的模板保持明文。
- 默认模型来自项目 allowlist，不允许直接调用被阻止的模型命名空间或任意未审查模型。

## 验证范围

直接测试覆盖：

- OpenRouter 模型策略、secret 加载、结构化输出、错误分类和一次修复；
- context 边界、历史反思、动作校验、Gateway stale/unknown 语义；
- 九节点执行顺序、checkpoint、trace 和 prompt budget；
- `PersonaAgent` 身份/Persona prompt、SABM 多公司 intent 汇总和 Controller fallback；
- `run_agents` 的默认 OpenRouter 参数和 Controller manifest。

完整交付还需运行后端全量 pytest、前端基线 lint/test/build、Windows 启动器验证，以及 git-crypt staged/committed blob 审计。
