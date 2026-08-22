# Provider 与全模式端到端验证报告

日期：2026-08-22

## 结论

OpenRouter 默认模型 `nvidia/nemotron-3-super-120b-a12b:free` 已通过真实结构化调用，并完成单、双、四 Agent 的 5 回合 PersonaAgent/SABM Episode。四 Agent 压力场景接受 16/20 个真实意图，其余 4 个按原 Controller 规则 fallback；没有把失败伪装成 accepted。后端全量 253 项、前端 11 项、lint、启动器 SmokeTest 和 git-crypt 审计均通过。

WebUI 已在用户可见的前台浏览器完成验收。三入口、全部配置、自定义 Persona/驱动、Agent 增删、导航、通信过滤、重建节点、完整 20 回合、报告解锁与下载资源均有可观察结果；390×844 视口无横向溢出。

## 真实 OpenRouter 场景

复现真实模型测试前，先按 [OpenRouter API Key 创建与本地配置](05-openrouter-api-key-setup.md) 完成本地密钥配置；测试和报告不包含密钥原文。

| Episode | 条件 | 结果 |
|---|---|---|
| `e2e-openrouter-statereset-v3-20260822` | 1 模型 Agent，balanced，perfect，5 轮 | 4/5 accepted，1 fallback |
| `e2e-openrouter-public-pair-20260822` | 2 模型 Agent，selfish_long_term / conservative，public，value_oriented | 5/10 accepted，5 fallback |
| `e2e-openrouter-four-agent-attempt-20260822` | 4 模型 Agent，4 Persona，public，quality_oriented | 16/20 accepted，4 fallback |

四 Agent 每家公司均产生 5 条 trace；accepted trace 都有 3 个候选，receipt 数与 accepted 数一致。失败分类仅为 `truncated` 或 `domain_validation_failed`，修复次数不超过 1。Persona manifest 分别记录 aggressive、conservative、profit_myopic、balanced。

行为方向总体合理：长期自利 Agent 的有效动作偏服务投入与价格调整，保守 Agent 的有效动作更偏韧性预算；模型并非机械单调行动。所有动作先通过 action contract、现金和状态哈希校验，市场份额由 Controller 统一归一并结算。

## 模式矩阵

使用 mock Provider 隔离验证市场和编排层，共 7 个 Episode、35 轮、140/140 个意图接受：

- 市场：random、balanced、value_oriented、quality_oriented、service_oriented；
- 信息：perfect、public；
- 通信：off、public_only、public_private；
- 四家公司、四种 Persona、每组 5 轮。

每组 state_version 从 1 连续递增到 5。OpenRouter SABM 仍明确限制为 `communication-mode=off` 和 `opponent-policy=controller-rule`；CLI 对不支持组合直接报错，不伪造通信、合作或博弈增强运行。

## 修复项

1. SABM Runner 原先自造 `sabm:<episode>:round:<n>` step ID，违反 Controller 契约并导致 422；改为读取权威 round/state_version。
2. LangGraph 同 thread 跨轮合并旧 `intent_receipt`；每轮显式清空全部轮内字段，并新增连续两轮回归测试。
3. SQLite checkpoint 对本地 Pydantic 类型发出未来阻断警告；改用精确的 msgpack 类型白名单。
4. Persona 进入 prompt 但未进入 trace；新增向后兼容的 `persona_manifest` 和 profile hash。
5. 前端新增与 `~member-2-v1` 一致的竖向八节点详情；每个节点展示职责、输入、输出和证据字段，DEMO 与未接入 trace 的真实 Episode 使用不同状态文案。
6. WebUI 移除 API Key/Token 输入与请求头；OpenRouter 固定从后端 `secrets` 读取，Coordinator 协同组合明确转交本地 Coordinator。
7. 响应式侧栏折叠后导航只剩序号；新增稳定 `aria-label`，窄屏仍可准确访问全部页面。
8. 报告“导出当前结果”原为死按钮；现使用原生下载链接导出只含可见汇总与证据边界的 JSON。

## 验证证据

- `python -m pytest`：253 passed，1 个既存 Starlette/httpx 弃用警告。
- `npm run lint`：通过。
- `npm test`：构建通过，11 passed。
- `scripts/start.ps1 -SmokeTest -NoBrowser`：API、Gateway、前端健康检查通过并清理服务。
- git-crypt audit：`real=1; examples=1; failures=0`；真实 secret 的 committed blob 保持 git-crypt 密文。
- 可见前台浏览器：基础真实 Episode 无凭据创建成功；Coordinator 组合不接收浏览器密钥；观察/研究演示完成 20 回合且不会循环；报告仅在完成后解锁。
- 竖向节点流：8 个节点、7 个向下连接符；桌面和 390×844 视口均为单列，窄屏 `scrollWidth=375 < innerWidth=390`。

## 剩余限制

- 前端没有 Job API 或 trace stream，因此真实 Episode 节点只显示拓扑和等待状态。
- OpenRouter 免费模型存在截断、空响应和领域校验失败，当前通过一次修复与 Controller fallback 保证 Episode 前进。
