# 源项目修改总结与不足

日期：2026-08-23

## 说明与边界

本文总结 `member-2` 工作在当前 `source` 仓库中对源项目所做的增量修改，以及仍未完成或仅部分接通的能力。市场环境、Controller 权威结算、原有 Agent Gateway 和既有 Provider 主体仍被保留；改动主要集中在 Windows 启动、固定模型凭据、OpenRouter Provider、PersonaAgent/SABM 后端决策流、WebUI 对接和真实运行可视化。

详细接口、失败语义与安全边界分别见：

- [PersonaAgent/SABM 后端设计](01-personaagent-sabm-backend-design.md)
- [AI 调用与生成过程实时信息设计](03-ai-live-progress-design.md)
- [Provider 与全模式端到端验证报告](02-provider-full-mode-e2e-validation-report.md)

## 已完成修改

### 1. Windows 一键启动

- 新增根目录 `start.bat`，调用 `scripts/start.ps1`。
- 自动准备 `.venv` 和前端依赖，释放目标端口，构建并启动 API、Agent Gateway 与 WebUI。
- 服务可用后可打开浏览器；当前开发验收均在用户可见的前台浏览器进行。

### 2. 固定模型凭据与 Provider

- WebUI 不再要求用户当场输入 API Key 或 Controller Token。
- OpenRouter Key 固定从 `secrets/open_router-api_key.env` 读取，仅在缺失或模型调用报错时显示对应排障信息。
- Key 的创建、本地配置、验证和泄露后更换流程统一见 [OpenRouter API Key 创建与本地配置](05-openrouter-api-key-setup.md)。
- `.gitattributes` 对 `secrets/**` 强制使用 git-crypt；`.example` 文件保持可公开明文。
- 新增模型 allowlist、结构化输出校验、错误分类、一次修复和安全 fallback；保留源项目原有 Doubao、DeepSeek 与 Mock 路径。
- WebUI 默认提供经过本地门控的免费模型选项，但其远端可用性不由本项目控制。

### 3. PersonaAgent 与 SABM 节点流

- 新增 `PersonaAgent`，把公司、Persona Profile、模型和单公司 Runtime 绑定。
- 新增有界 SABM/LangGraph 决策流：读取快照、构建上下文、策略反思、生成候选、校验/修复、准备 Intent、提交 Intent 和完成审计。
- 多家公司并发运行，只有 accepted Intent 进入 Controller；失败公司继续使用源项目的规则 fallback。
- checkpoint 和 trace 按 Episode/公司隔离，修复了跨轮临时状态污染、step ID 和本地 checkpoint 类型恢复问题。
- trace 记录安全的记忆视图、决策输入、策略反思、结构化候选、选择结果、Token、耗时、修复、Intent 回执和终态，不返回密钥、Provider 原始响应或隐藏思维链。

### 4. WebUI 控制与模型配置

- 控制方式统一为“人类参与者”或“AI 模型”。
- 参与者模式默认公司 A 由人类控制，其余三家为 AI；观察者和研究员模式默认四家均为 AI。
- 新实验默认 5 回合，可按公司选择固定模型和 Persona。
- 基础真实回合使用 `POST /api/episodes/{episode_id}/managed-rounds`，由后端统一运行模型、接收人类动作并调用 Controller 联合结算。
- 正常 UI 不显示 Provider 名称；Provider 专用信息仅用于密钥缺失或 AI 错误排障。

### 5. 真实数据与演示数据隔离

- 新增 `frontend/app/real-runtime.ts` 作为真实 Episode 的独立投影，源码测试禁止其引用 `DEMO_*`。
- 参与者、观察者和研究员三种真实入口共用同一数据边界。
- AI 未运行前不再预填消息、信念、策略、动作、市场结果、图表或回放。
- AI 运行后只使用当前 Episode 的 observation、execution trace、Intent resolution、Controller settlement 和 append-only history。
- `03` 至 `06` 的真实模式分别显示实际通信空状态、实际策略 trace、Controller 经营结果和本次 Episode 回合记录。

### 6. AI 后端节点流可视化

- 完整节点流放在 `02 智能体观察`；参与者模式三家 AI 三列并排，每列内部纵向展示八个节点。
- 观察者和研究员模式展示四家公司，可横向查看；窄屏退化为单列。
- `01 实时现场` 只显示紧凑 AI 状态，避免把产品六步说明与后端节点事件混在一起。
- 终态节点可查看后端返回的完整安全 trace，包括实际决策上下文、候选、选择、校验、Intent 与回执。

### 7. AI 调用过程实时信息

- 新增线程安全的 `managed_round_progress.py`，按 Episode 和公司隔离进度。
- 新增 `GET /api/episodes/{episode_id}/managed-rounds/progress` 只读接口。
- 前端在真实管理回合运行期间每 500ms 轮询；切换 `01/02` 不会停止轮询。
- `01` 显示当前节点、模型等待、调用次数和实际已用时间；Provider 返回后才显示真实 Token、模型耗时、修复、错误类别与 fallback。
- `02` 根据真实事件显示“执行中 → 已执行 / 错误”，不使用预计百分比或固定动画阶段。
- 回合完成后，完整 execution trace 替换临时进度，Controller 响应仍是最终权威结果。

### 8. 测试、文档与本地提交

- 增加 Provider、PersonaAgent、SABM、managed round、真实数据隔离、进度注册表、轮询和节点状态测试。
- 已对参与者、观察者、研究员三种模式执行真实模型首轮可见浏览器测试。
- 相关工作由连续本地 Git commit 记录；最近的关键提交包括：

| Commit | 内容 |
|---|---|
| `fb7d16e` | PersonaAgent/SABM Provider 流程验证 |
| `9fa5567` | 固定 secrets 与竖向节点流 |
| `0934b79` | WebUI 详细 AI 节点流 |
| `3f15682` | 真实数据与演示数据严格隔离 |
| `1d600a9` | AI 调用与生成过程实时进度 |

## 关键文件索引

| 路径 | 职责 |
|---|---|
| `start.bat` / `scripts/start.ps1` | Windows 一键启动 |
| `src/game_theory_agent/agents/single/` | Provider、PersonaAgent、SABM 单 Agent Runtime 与 trace |
| `src/game_theory_agent/api.py` | WebUI Episode、managed round、Controller 对接与进度 API |
| `src/game_theory_agent/managed_round_progress.py` | 线程安全的实时进度注册表 |
| `frontend/app/real-runtime.ts` | 真实 Episode 独立前端投影 |
| `frontend/app/managed-round-progress.ts` | 真实进度轮询与临时 execution 映射 |
| `frontend/app/sabm-node-flow.ts` | 八节点状态与详情映射 |
| `frontend/app/sabm-node-flow-view.tsx` | 节点流展示组件 |
| `frontend/app/page.tsx` | 三入口配置、回合运行和各工作区组合 |
| `tests/test_managed_ai_web.py` | managed round 与进行中进度 API 测试 |
| `frontend/tests/` | 控制方式、真实数据隔离、轮询、节点流与报告测试 |

## 当前不足与待完成

### P0：影响核心功能完整性

#### 1. 未支持通信

源项目已有通信协议和相关数据结构，但 WebUI 的 managed round 明确要求 `communication_mode=off`。当前 `03 通信记录` 在真实基础回合中通常只能显示“0 条消息”，尚未完成 AI 消息生成、公开/私信屏障、消息关闭和结算前同步。

待完成：把通信阶段纳入管理回合生命周期，严格区分公开消息与公司私信，并将实际通信事件接入 `03`、`02` 和 Replay。

#### 2. 未支持 **共享抗冲击投入**

人类动作表单保留共享投入字段，源项目也有 Cooperation/Shared Resilience 机制，但 WebUI 管理回合当前要求 `cooperation_mode=off`，因此尚未真正运行提议、承诺、接受/拒绝、实际贡献、履约或背离流程。

待完成：接入 Cooperation 屏障协议、贡献结算、公共韧性变化、履约审计和合作 Replay；避免把非约束承诺当作已执行动作。

#### 3. 未支持 **博弈分析辅助**

源项目已有 Belief、Opponent Model、Utility Inference、Bayesian Advisor 和 Repeated Game 模块，但 WebUI 的真实管理回合尚未运行这些处理条件。`04 信念与策略` 当前主要展示 PersonaAgent trace 中已有的策略反思、候选和选择，不等同于完整博弈分析辅助。

待完成：按回合接通 Belief 更新、对手模型、效用推断、Advisor 候选和采纳差异，并保留 treatment/off 对照与 Advice Hash/Replay。

#### 4. 三项高级能力尚未形成统一回合屏障

通信、合作和博弈分析不是三个独立开关即可完成；它们需要在 Observe → Communication → Belief/Strategy → Intent → Controller Settlement 之间建立确定顺序、超时、fallback 和幂等语义。当前基础 managed round 只覆盖无通信、无合作的 AI/人类 Intent 与结算。

### P1：影响可读性、研究可用性与稳定性

#### 5. 未可视化解析 JSON

**AI 后端节点流**和**回合记录**仍大量使用 `<pre>` 展示 JSON。虽然数据真实且详细，但用户需要自行阅读字段、嵌套对象、金额单位和状态码，难以快速比较三家/四家模型。

待完成：为 Observation、候选策略、最终动作、Token/耗时、验证错误、Intent 回执、Controller settlement 和 Replay 增加字段化卡片、折叠树、差异高亮、单位格式化与原始 JSON 备用视图。

#### 6. UI 未优化整理

当前 UI 已能运行核心流程，但 `frontend/app/page.tsx` 同时承担入口、配置、请求、状态和多个工作区，职责过重；部分页面信息密度高、字号偏小、卡片层级重复，四模型节点流主要依赖横向滚动。

待完成：按入口、配置、回合运行、节点观察、市场、Replay 和报告拆分组件；统一空状态、状态色、字号档位、间距和移动端布局，并对长 trace 使用虚拟化或按需展开。

#### 7. 实时进度只保存在内存

最新 managed round 进度保存在 API 进程内。页面切换可以继续读取，但后端重启后进度丢失；它也不是完整的历史事件存储。

待完成：为运行中的 Job 增加持久化状态或从 append-only trace 重建，并定义服务重启后的 running → interrupted 恢复语义。

#### 8. 缺少回合取消、恢复和失败重试

当前进度接口只读，用户不能取消长时间模型调用。页面刷新后可以读取后端最新进度，但前端无法恢复原 POST Promise；Provider 暂时失败时主要依赖一次修复和规则 fallback。

待完成：增加安全取消、超时、幂等重试、刷新后状态恢复和“继续下一轮”规则，确保取消不会提交过期 Intent。

#### 9. 免费模型稳定性不足

免费模型可能限流、下线、变更上下文限制或返回非结构化内容。当前使用固定 allowlist 和人工探测结果，没有启动时健康检查、动态降级顺序或可用性缓存。

待完成：增加脱敏健康检查、模型状态缓存、明确的备用模型策略和 UI 可用性提示；切换模型必须写入 manifest 与 trace。

#### 10. 参与者隐私投影仍可进一步收紧

当前本地 API payload 面向同一可信进程，参与者、观察者和研究员共享较宽的 Episode 响应，再由前端选择展示范围。若扩展为多用户或远程部署，应在后端按角色裁剪 observation、公司私有财务和 trace，而不能只依赖 UI 隐藏。

待完成：增加角色化响应 DTO、公司级授权和隐私回归测试，确保参与者永远拿不到对手私有字段。

### P2：影响研究规模与产品化

#### 11. 缺少三模式完整 5 回合真实回归

当前自动化覆盖较完整，三模式也已完成首轮真实模型冒烟，但尚未固定执行参与者、观察者和研究员各自完整 5 回合的长期状态、跨轮 checkpoint、终局报告和 Replay 验收。

待完成：建立可重复的 3 模式 × 5 回合端到端矩阵，检查跨轮记忆、市场份额守恒、终局、报告和浏览器控制台错误。

#### 12. 缺少多实验对照与统计分析

研究员模式尚未提供批量 Seed、处理组/对照组、配对运行、失败重跑、指标聚合、置信区间和实验差异可视化。单个 Episode 只能形成描述性结果。

待完成：增加实验队列、共同随机种子配对、处理条件 manifest、批量导出、统计汇总和证据等级标记。

#### 13. 回合记录缺少可搜索审计工具

当前 `06 回合记录` 能查看本次 Episode 的真实 state、resolution 和 execution，但仍以单次响应 JSON 为主，缺少按公司、节点、错误码、Intent、hash 和时间过滤。

待完成：增加时间线、公司/节点过滤、状态差异、hash 校验结果、事件跳转和导出选择。

#### 14. 本地单机假设较强

当前 Controller、Gateway、进度注册表和 WebUI 主要面向本机单用户环境，尚未提供登录、租户隔离、权限管理、审计保留策略、速率限制和远程部署方案。

待完成：在确有远程部署需求后，再设计认证授权、持久化任务队列、数据库、对象存储和运维监控；当前阶段不提前引入这些复杂度。

#### 15. 可观测性仍不完整

当前已有节点事件、Token、耗时和安全错误类别，但缺少跨 Episode 指标、模型成功率、fallback 比例、P50/P95 延迟、成本估算和统一运行仪表板。

待完成：在不记录 prompt/原始响应和密钥的前提下，增加聚合指标与本地诊断报告。

## 推荐后续顺序

1. 先完成 JSON 字段化可视化和 UI 组件拆分，降低后续高级能力接入的页面复杂度。
2. 接入通信回合屏障，并完成真实消息、隐私和 Replay。
3. 接入共享抗冲击投入，完成承诺、贡献、结算和履约审计。
4. 接入博弈分析辅助，建立 treatment/off 对照与可解释差异。
5. 补齐取消/恢复、持久化进度和免费模型健康检查。
6. 建立三模式完整 5 回合回归，再扩展批量 Seed、统计分析和研究报告。
7. 只有在需要远程多人使用时，再实施角色授权、多租户和生产部署架构。

## 当前结论

本次修改已经把源项目的既有市场与 Controller 核心，接到固定模型 Provider、PersonaAgent/SABM 节点流和可见 WebUI 真实回合中，并解决了演示数据混入与长模型调用不可观察的问题。当前可视为“基础真实 AI 回合与审计可视化已跑通”，但还不是包含通信、合作、完整博弈分析、结构化 Replay 和批量研究能力的完整实验平台。
