# Changelog

## 2026-08-23

- 新增 AI 回合真实进度接口与 WebUI 轮询：运行期间显示每家公司当前 SABM 节点、模型等待、调用次数和实际耗时；Provider 返回后才显示 Token、模型耗时、修复、错误与 fallback。
- `02 智能体观察` 的三列/四列节点流支持“执行中 → 已执行 / 错误”实时状态，页面切换不终止回合轮询，最终由完整 execution trace 替换临时进度。
- 参与者、观察者、研究员三种真实模式统一改用独立的后端数据投影；模型运行前不再预填消息、信念、策略、动作、结算、图表或回放，首次结算后只展示本次 Episode 的 observation、execution trace、Controller 状态与历史。
- 完整 PersonaAgent/SABM 节点详情移至 `02 智能体观察`；`01 实时现场` 只显示每个 AI 的紧凑进度，`03` 至 `06` 分别使用实际消息、策略 trace、经营结果与回合记录。
- Episode API 增加公司 observations 与 append-only history，使前端初始态和回合重建均可追溯到权威后端响应。
- AI 后端节点流改为三家模型三列并排、每列八节点纵向展示，不再复用六步说明的公司选择状态。
- 节点详情对齐 `~member-2-v1`：安全展示真实记忆视图、实际决策输入、策略反思、三候选及选中理由、校验/修复、待提交意图、提交回执和终态；仍不返回密钥、Provider 原始响应或隐藏推理。

## 2026-08-22

- WebUI 控制方式收敛为“人类参与者 / AI 模型”，支持按参与者选择实测通过的 Super 或 Ultra 免费模型；默认实验改为 5 回合。
- 新增 WebUI 后端管理回合：真实运行 PersonaAgent/SABM、合并人类 Intent、联合结算，并把节点事件、模型、耗时、Token、修复、错误与 fallback 安全返回前端。
- 竖向节点流取消静态输入/输出说明和 DEMO 假进度；正常 UI 隐藏 Provider 名称，仅排障错误可以显示。
- AI 后端节点流与六步交互说明解耦，使用独立的 AI 公司区域，避免把人类公司的“接收信息”误作模型节点事件。
- 修复 SABM 结算 step ID、跨轮 checkpoint 临时状态污染和 Persona trace 缺失；为本地 checkpoint 类型增加显式反序列化白名单。
- 实测 OpenRouter 单 Agent、双 Agent和四 Agent 真实回合；新增五种市场、两种信息和三种通信模式回归矩阵。
- 前端新增只读 PersonaAgent/SABM 八节点纵向详情视图和安全 JSON 报告导出；真实 Episode 未接入 trace 时不伪造节点进度。
- 移除 WebUI 的密钥输入与请求头；OpenRouter API Key 固定由后端 `secrets` 读取，基础真实实验直接创建，高级组合明确交由本地 Coordinator。
- 新增 Windows `start.bat` / `scripts/start.ps1` 一键启动入口。
- 新增 git-crypt 保护的 OpenRouter secrets、本地 OpenRouter Provider、`PersonaAgent` 类和 SABM 后端节点流；保留原有 Provider，未修改前端和公共 API。
- 新增 `docs/member-2/` 后端设计与对接说明。
- 将项目文档按基础规范、Agent 平台、信息与博弈、Persona、交互与合作五个主题分类。
- 为主题目录和目录内文档增加连续编号，并新增统一文档索引。
- 更新 README、CONTRIBUTING 和文档间引用以匹配新路径。
