# 协作开发约定

本仓库由四人共同开发。默认分支 `main` 应始终保持可运行，功能通过短生命周期分支和 Pull Request 合入。

## 开始开发

```powershell
git clone https://github.com/Robert-WDY/LLM-Guided-Multi-Agent-Strategic-Decision-Making-in-Dynamic-Competitive-Markets.git
cd LLM-Guided-Multi-Agent-Strategic-Decision-Making-in-Dynamic-Competitive-Markets
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env

cd frontend
npm ci
Copy-Item .env.example .env.local
```

不要在 `.env.example` 中放真实令牌；个人密钥只保存在被忽略的 `.env` 或 `.env.local`。

## 分支与 Pull Request

- 分支名使用 `feature/<topic>`、`fix/<topic>`、`docs/<topic>` 或 `experiment/<topic>`。
- 一条分支只处理一个主题，避免同时大改市场公式、API Schema 和前端布局。
- 开 PR 前先同步 `main`，解决冲突并完成本地测试。
- 至少一名非作者成员 Review 后再合并；市场公式、接口 Schema 和 Seed 协议建议由对应模块负责人 Review。
- 推荐 Squash merge，让 `main` 中每个提交都对应一个完整变化。

提交信息建议采用 `type(scope): summary`，例如：

```text
feat(market): distinguish quality and service utility
fix(api): preserve configurable episode horizon
docs(agent): document controller trust boundary
```

## 模块边界

| 区域 | 主要路径 | 修改要求 |
| --- | --- | --- |
| 市场模型 | `src/game_theory_agent/market/`, `configs/` | 参数、实现、测试和版本号同步修改 |
| 决策与 Agent | `decisioning.py`, `gameplay.py`, `api.py` | 保持 Agent 只提交意图、Controller 统一执行 |
| 前端 | `frontend/app/` | 不复制后端市场公式，数据口径来自 API |
| 验证与文档 | `tests/`, `docs/` | 新行为必须有确定性测试和公开字段说明 |

建议四位成员各自维护一个主要区域，但关键改动仍需交叉 Review。确认 GitHub 用户名后，再添加 `.github/CODEOWNERS` 固化负责人；不要用猜测的账号创建错误规则。

## 合并前检查

```powershell
python -m pytest

cd frontend
npm run lint
npm run build
```

如果修改市场随机过程，还应运行 200 Seed 校准，并用固定 Seed 做新旧版本配对测试。提交中不得包含 `node_modules`、构建输出、缓存、日志、运行时 PID、真实密钥或本地实验临时文件。

## 兼容性要求

- 更改 State、Action、Event 或 Manifest 结构时提升相应 Schema/环境版本。
- 已公开的 Agent 字段不得静默改变语义；需要迁移时在 `docs/02-Agent平台与控制台/01-agent-gateway-api.md` 说明。
- 保证相同配置、Seed、初始状态、动作和环境版本产生相同 State Hash。
- 配置参数以 `configs/market_v4.yaml` 为唯一来源，避免在前后端重复硬编码市场公式。
