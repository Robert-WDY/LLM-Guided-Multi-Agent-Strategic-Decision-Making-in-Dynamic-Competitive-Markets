# 2026-09-11 本机 Git 与迁移 ZIP 融合记录

## 范围与来源

用户本轮要求：先拉取最新 Git，再把另一台电脑的 ZIP 更新融合进 `C:\Users\asus\Desktop\game-theory-agent`。交接文档作为历史参考；没有据此执行新增研究、付费模型实验、发布或推送。

- 目标目录最初没有 `.git`，保留的是较早的 v10 源码和旧电脑依赖环境。
- 先通过本机已启用的代理 `127.0.0.1:7897` 克隆远端，再将 Git 元数据与最新文件整合到指定目录，最后执行 `pull --ff-only` 确认最新。
- 远端：`https://github.com/Robert-WDY/LLM-Guided-Multi-Agent-Strategic-Decision-Making-in-Dynamic-Competitive-Markets.git`。
- 分支：`main`；本轮核验的 HEAD / origin/main：`192c0fdbffe8f429d4c195d940db78aa39882481`。
- ZIP 基线与远端相同，ZIP 中的 10 个已跟踪文件改动与 `git-status.txt` 一致；新增 v20/v20.1 建议器、配置、测试、脚本、报告和历史实验资料已导入。
- ZIP：`C:\Users\asus\Desktop\洞墟过渡\gametheory\gametheory-conversation-code-experiments-20260909-150725.zip`。
- ZIP SHA-256：`4611f1ba9ed313a9983fe5c9f34cec69c9811949ebb500ca96fac8578da56a7f`，与交接值一致。

## 合并与保护

新增 15,668 个文件，更新 37 个已有文件；所有更新前的原文件均复制到 `C:\Users\asus\Desktop\codex-workspace\gametheory-integration-20260911\local-before` 并核对内容。未被新版本覆盖的本机独有文件保持原位，包括 `.env` 和用户文档。较早研究日志是 ZIP 研究日志的完整前缀。

远端中存在但 ZIP 未收录的文件也保留。归档缺失不当作删除指令。没有代码冲突需要人工取舍；旧版差异对应后续已演进的实现。

完整导入清单及原文件、导入文件 SHA-256 位于 `C:\Users\asus\Desktop\codex-workspace\gametheory-integration-20260911\integration-files.json`；原始 ZIP 清单、Git 差异和历史保存在同目录的 `archive` 中。

适配 `AGENTS.md` 的日志镜像路径为 `../codex-workspace/local-history/PROJECT_EVOLUTION_RECORD.md`。研究日志、密钥和本机用户文件通过 `.gitignore` / `.git/info/exclude` 排除。修正了跨目录复制 Git 索引后产生的行尾状态缓存；没有改写归档证据字节，也没有暂存业务改动。

交接文档原文已复制到 `docs/PROJECT_HANDOFF_2026-09-11.md`。本次改动保留在工作区，未创建提交或推送。

## 验证

- ZIP 解压时逐文件核验全部 17,484 条清单记录的大小和 SHA-256，全部通过；融合后再次核验全部文件，除本轮有意适配的 `AGENTS.md` 和追加的研究日志外，归档内容保持一致。
- v20.1 `source.zip` SHA-256：`c1433e48b091939b3fe79a3deb77f3a94475f2ce13aefeae839aba18f10e6433`。
- v20.1 `evidence.zip` SHA-256：`7263b354b6aea8a7f90f00367e168a194cd21589d1ce9af75232062119a77bd5`。
- 原 `.venv` 引用了 `C:\Users\12204` 下的 Python，无法使用；已将其移至备份目录，并用本机 Python 3.12.14 重建 `.venv`，安装 `.[test]`，`pip check` 通过。
- 现有前端依赖与当前版本可兼容；Node.js 24.19.0 下生产构建、TypeScript 类型检查、ESLint、19 项前端测试全部通过。
- Python 全量回归：530 项全部通过，耗时 171.17 秒；1 条第三方 Starlette/AnyIO 弃用提示，无失败。
- `git diff --check` 通过，暂存区为空。

验证日志位于 `C:\Users\asus\Desktop\codex-workspace\gametheory-integration-20260911`：`pytest.log`、`pytest-results.xml`、`frontend-build.log`、`frontend-tests.log`。前端通过 `node node_modules/vinext/dist/cli.js build` 及对应 Node/TypeScript/ESLint 入口验证；当前会话没有可用的 `npm` 命令，未声称交接中的 npm 启动脚本已执行通过。

## 尚未迁移的运行状态

ZIP 不含费用账本和旧 UI 会话数据库，目标目录也没有原 `model-budget.sqlite3` / `model-budget.initialized`。未初始化费用账本、未启动本地服务、未进行真实模型调用。后端测试会话数据库定向到仓库外的 `test-state`，避免创建或覆盖正式运行会话。

工程迁移和回归不改变既有研究结论：v20/v20.1 的经济可靠性仍未全部通过，福利覆盖率、尾部风险和长期收益保证仍有缺口。恢复旧费用账本及旧会话需要另外的数据来源。
