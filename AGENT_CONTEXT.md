# Agent Context

## 项目结构

- `src/game_theory_agent/`：市场引擎、Controller/Gateway、Agent 编排与 Provider。
- `frontend/`：React/Vinext WebUI；无后端时只运行明确标注的 DEMO。
- `tests/`：后端确定性、权限、编排和 Provider 测试。
- `docs/member-2/`：OpenRouter PersonaAgent/SABM 专项设计、对接和验证报告。

## 架构约定

- 8010 是受保护 Controller，8011 是 Agent Gateway；二者必须由同一 Python 进程承载内存 Episode。
- OpenRouter PersonaAgent 只经 Gateway 读取公司视图和提交 intent；SABM Runner 使用 Controller 权威 round/state_version 结算。
- 单 Agent LangGraph checkpoint 可跨轮保存，但所有轮内临时字段必须在 `decide_round` 开始时重置；实验历史从 append-only trace 读取。
- secrets 工作树可为明文，Git index/commit 必须由 git-crypt 加密；任何报告不得包含 key、Provider 原始响应或隐藏推理。节点审计可以展示后端标记为安全的 `prompt_audit`。
- 基础 WebUI 真实回合调用 `/api/episodes/{episode_id}/managed-rounds`；后端并发运行 PersonaAgent、提交人类 Intent、执行 Controller 联合结算，并只返回节点图需要的安全 trace 字段。
- 正常 UI 不显示 Provider 名称；只有密钥缺失或 AI 调用错误允许显示排障信息。未运行节点必须显示“尚未运行”，不得从前端步骤生成假进度。AI 后端节点流按 AI 公司三列并排，每列八节点纵向展示结构化安全 trace，不得混入人类公司的六步说明。
- 真实模式只能通过 `frontend/app/real-runtime.ts` 投影后端 `state / observations / decision_resolutions`；该模块不得导入 `DEMO_*`。三种入口共享这条边界，初始态不产生消息、信念、计划、决策、动作、结算或历史。
- `01` 仅显示运行进度，完整节点详情位于 `02`；`03` 至 `06` 只消费本次 Episode 的实际通信、execution trace、Controller 结果和 history。演示数据只能在显式交互演示分支使用。
- AI 回合运行时，`managed_round_progress.py` 保存按 Episode/公司隔离的安全节点事件；`GET /api/episodes/{episode_id}/managed-rounds/progress` 只读返回最新快照。前端仅在真实 POST 等待期间每 500ms 轮询，切换导航不停止；终态必须以 POST 返回的完整 execution trace 为准。
- 实时进度不得包含 prompt、候选正文、原始响应、凭据、预计百分比或预计剩余时间。Token 与模型耗时只能在真实 provider response/error 事件产生后显示。

## 验证

- 后端：`.\.venv\Scripts\python.exe -m pytest`
- 前端：`npm run lint`、`npm test`
- 启动器：`pwsh -File scripts/start.ps1 -SmokeTest -NoBrowser`
- secrets：使用固定 `managing-secrets` 工具链执行 git-crypt audit。

## 已知问题

- WebUI 管理回合当前只支持 `communication_mode=off`、`cooperation_mode=off`；高级交互组合仍由本地 Coordinator 负责。
- Starlette TestClient 仍有 httpx 迁移弃用警告。
