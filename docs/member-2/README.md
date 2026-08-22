# Member 2 后端对接说明

本目录说明 OpenRouter Provider、`PersonaAgent` 类、SABM 节点流后端接入及 WebUI 真实节点拓扑。基础 WebUI Episode 使用本地管理回合 API；不向浏览器暴露 Provider 或 Controller 凭据。

源项目增量修改、关键文件、当前不足和推荐后续顺序见 [源项目修改总结与不足](04-source-modification-summary-and-gaps.md)。

## 使用边界

- 后端入口：`python -m game_theory_agent.run_agents`。
- 默认 Provider：`openrouter`；仍可显式选择 `doubao`、`deepseek` 或 `mock`。
- OpenRouter secret：`secrets/open_router-api_key.env`，工作树为本地明文，Git blob/commit 由 git-crypt 加密。
- 首次使用前，按 [OpenRouter API Key 创建与本地配置](05-openrouter-api-key-setup.md) 创建密钥、复制模板并验证模型连接。
- WebUI 默认模型：`nvidia/nemotron-3-super-120b-a12b:free`；实测通过的复杂推理备选为 `nvidia/nemotron-3-ultra-550b-a55b:free`。
- 当前 SABM 接入要求 `--communication-mode off` 和 `--opponent-policy controller-rule`；未提交 intent 的公司由 Controller 原有规则回退处理。
- checkpoint 与 trace 默认写入 `~outputs-intermediate/agent-runs/`，不进入 Git。

## 启动

Windows 可从项目根目录双击或执行：

```powershell
.\start.bat
```

脚本调用 `scripts/start.ps1`，使用/创建 `.venv`、按 `requirements.txt` 补齐依赖、停止占用 3210/8010/8011 的监听进程、启动后端与前端，并在服务可用后打开浏览器。

运行 OpenRouter PersonaAgent/SABM 后端前设置 Controller token：

```powershell
$env:MARKET_CONTROLLER_TOKEN="请替换为随机高熵令牌"
$env:PYTHONPATH="src"
python -m game_theory_agent.run_agents --rounds 5 --seed 42
```

指定多家公司和 Persona：

```powershell
python -m game_theory_agent.run_agents `
  --provider openrouter `
  --agent-companies company_A,company_B `
  --persona balanced `
  --persona-map company_A=selfish_long_term,company_B=conservative `
  --rounds 5 `
  --seed 42
```

后端结构、节点流、失败语义和测试边界见 [PersonaAgent/SABM 后端设计](01-personaagent-sabm-backend-design.md)。
AI 回合执行期间的真实节点事件、进度轮询、安全字段和三模式展示设计见 [AI 调用与生成过程实时信息设计](03-ai-live-progress-design.md)。

实现使用 `GET /api/episodes/{episode_id}/managed-rounds/progress` 返回当前 Episode 的安全进度快照。状态按公司隔离，包含真实节点事件、调用次数、后端计时和 Provider 已返回的用量指标；不包含 prompt、候选正文、原始响应或凭据。

WebUI 每轮调用 `POST /api/episodes/{episode_id}/managed-rounds`。请求只包含各人类公司的结构化动作；后端从 Episode manifest 读取每家 AI 的固定模型和 Persona，并发运行节点流，再使用 Controller 的 `fallback=rule` 联合结算。响应中的 `executions` 按公司提供真实节点事件、后端标记为安全的完整 `prompt_audit` 与结构化 trace；三家 AI 在前端按三列并排展示。响应绝不包含密钥、Provider 原始响应或隐藏推理。

真实模式的 Episode 响应同时包含各公司的实际 `observations` 和 append-only `history`。参与者、观察者与研究员共用不依赖演示常量的投影：运行前只显示权威初始状态，运行后再填入真实 trace、resolution、settlement 和 replay。完整节点流在 `02 智能体观察`，`01` 只显示运行状态。
真实 Provider、模式矩阵、缺陷修复和剩余可见浏览器限制见 [Provider 与全模式端到端验证报告](02-provider-full-mode-e2e-validation-report.md)。
