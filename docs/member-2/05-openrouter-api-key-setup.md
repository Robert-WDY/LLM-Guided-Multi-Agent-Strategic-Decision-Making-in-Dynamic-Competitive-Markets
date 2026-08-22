# OpenRouter API Key 创建与本地配置

本项目不在 WebUI 中填写或传输 API Key。后端固定从项目根目录的 `secrets/open_router-api_key.env` 读取，前端无法获取密钥原文。

## 1. 创建 API Key

1. 登录 [OpenRouter API Keys](https://openrouter.ai/settings/keys)。
2. 选择创建新 Key，设置便于识别的名称；如果账户界面提供额度或到期时间，建议按实验需求设定限制。
3. 创建后立即复制密钥，不要发送到聊天、Issue、PR 或截图中。OpenRouter 官方说明创建接口只在返回时显示一次密钥原文，参见 [Create a new API key](https://openrouter.ai/docs/api/api-reference/api-keys/create-keys)。

## 2. 配置项目密钥

在 `D:\CODE\PY\market-agents\source` 中执行：

```powershell
Copy-Item .\secrets\open_router-api_key.env.example .\secrets\open_router-api_key.env
notepad .\secrets\open_router-api_key.env
```

把模板中的占位符替换为新 Key。文件支持以下任一格式，只选一种：

```text
sk-or-v1-REPLACE_WITH_YOUR_OPENROUTER_API_KEY
```

```text
OPENROUTER_API_KEY=sk-or-v1-REPLACE_WITH_YOUR_OPENROUTER_API_KEY
```

保存后可直接运行 `start.bat`。启动脚本和后端都使用上述固定路径，无需在浏览器再次输入。

## 3. 验证

可在项目根目录运行模型探测：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe .\scripts\probe_openrouter_models.py
```

脚本只输出状态、错误分类、耗时和 Token 指标，不输出密钥或模型原文。如提示密钥缺失，确认文件名精确为 `open_router-api_key.env`，且没有被记事本额外加上 `.txt`。

## 4. 安全与更换

- 真实密钥只保存在 `secrets/open_router-api_key.env`；远程 PR 只包含 `.example` 模板。
- 不要将密钥改写进 README、测试、日志、命令行参数或前端环境变量。
- 如密钥曾出现在提交、聊天或截图中，立即在 OpenRouter 密钥页面禁用/删除旧 Key，创建新 Key 并替换本地文件；不要只删除当前文件后继续使用旧 Key。
