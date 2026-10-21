# 新电脑交接：多智能体市场项目

更新日期：2026-09-11。

本文记录历史状态和迁移步骤。新对话以用户当前请求为准；本文不构成扩大预算、强推 Git 或对外部署的授权。

## 1. 目标与当前状态

项目用于个人本机运行，包含企业、消费者、供应商、政府自主决策，以及人格、多智能体合作竞争、博弈建议、保存恢复和实验评估。用户要求逐阶段实现、逐阶段独立实验验收，保留负结果并给出结论。

当前完成到 **v20 / v20.1 Advisor 可靠性修复与实验**。工程验证通过，但经济可靠性尚未全部通过。不能表述为建议普遍可靠、现实市场验证完成、全局最优或所有功能全面验收成功。

交接聚焦市场项目，不处理之前的 PDF。

## 2. 迁移资料

主归档：`gametheory-conversation-code-experiments-20260909-150725.zip`

- 旧电脑位置：`C:/Users/dongxu/Desktop/temp/output/`
- 大小：1,030,181,136 字节，约 982 MB。
- 包含 17,484 个项目文件，ZIP 中的 `project/` 是项目根目录。
- 包含完整当前源码、未提交修改、历史实验、发布包、报告、本地研究日志和文件校验清单。
- 包含从其他电脑迁移的早期资料，不表示所有内容均为本次对话原创。
- 不包含 `.git`、Python/Node 依赖环境、真实密钥或运行数据库。

主包 SHA-256：

```text
4611f1ba9ed313a9983fe5c9f34cec69c9811949ebb500ca96fac8578da56a7f
```

**费用账本需要单独迁移，主包没有包含：**

```text
<旧项目根目录>/.local-state/model-budget.sqlite3
<旧项目根目录>/.local-state/model-budget.initialized
```

应使用 SQLite 一致性备份，并核验记录。新电脑已有更新账本时不得覆盖回退。本次只生成本文，没有生成费用账本备份或补充 ZIP。

主包也没有旧 UI 会话数据库；`runs/` 的实验资料完整保留，但不能据此承诺恢复原浏览器会话。需要会话时应从旧机另外备份。

旧项目根目录：

```text
C:/Users/dongxu/Desktop/temp/gametheory/game-theory-agent/LLM-Guided-Multi-Agent-Strategic-Decision-Making-in-Dynamic-Competitive-Markets
```

新电脑可以选择短路径，例如 `D:/work/gametheory/project`。以下项目文件路径均相对于新项目根目录；不要照搬旧用户名、盘符或微信附件路径。

## 3. 首先阅读

1. `AGENTS.md`
2. `PROJECT_EVOLUTION_RECORD.md`
3. `docs/reliable-v20-results.md`
4. `docs/reliable-v20-error-isolation.md`
5. `docs/reliable-v201-plan.md`、`docs/reliable-v20-plan.md`
6. `docs/reliable-advisor-results.md`
7. `docs/market-evaluation-v18-results.md`

研究日志是本地资料，不提交 Git/GitHub。功能、实验、结论或计划变化时，应同步更新根日志和 `../../local-history/PROJECT_EVOLUTION_RECORD.md`，核对 SHA-256。如改变新电脑目录布局，应相应修订 `AGENTS.md` 中的镜像路径。

## 4. Git 状态

远端：

```text
https://github.com/Robert-WDY/LLM-Guided-Multi-Agent-Strategic-Decision-Making-in-Dynamic-Competitive-Markets.git
```

分支 `main`。最后核验的本地提交：

```text
192c0fdbffe8f429d4c195d940db78aa39882481
Add draft-based advisor and v19 evaluation evidence
```

该提交此前已推送；本次没有重新联网核验远端。用户要求的提交身份为 `Robert-WDY <robertdwdy@gmail.com>`。

**v20 / v20.1 修改尚未提交和推送，只 clone 会漏掉最新修复。** 新电脑需要 Git 工作流时先 clone，检查他人更新，再合并主包源码并审查 diff。保留新仓库 `.git`，不要强推或覆盖他人新修改。主包含 Git 状态、历史和差异文件，未跟踪新文件以 `project/` 实际内容为准。

新电脑还需在 `.git/info/exclude` 排除 `PROJECT_EVOLUTION_RECORD.md`。

主要变化包括：

- `src/game_theory_agent/game_theory/calibrated_advisor.py`
- `src/game_theory_agent/game_theory/empirical_advisor.py`
- 同目录 `service.py`
- `frontend/app/theory-lab.tsx`、`frontend/app/reliable-advisor.tsx`
- `configs/reliable-calibration-v20.json`、`configs/reliable-calibration-v201.json`
- v20/v20.1 实验、分析、滚动、独立回放和发布脚本。
- 建议器、API、保存恢复、预算日期测试及最新报告。

完整清单见主包 `git-status.txt`。

## 5. 新电脑安装和启动

安装 Python 3.12、Node.js >=22.13.0、Git。在项目根目录的 PowerShell 执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"

Push-Location frontend
npm ci
Pop-Location
```

首次启动前，先把核验过的费用账本及初始化标记恢复到项目 `.local-state/`。启动脚本会初始化不存在的账本，不能让新电脑因此丢掉历史费用和 30 元授权。

构建、启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_local.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/local_market.ps1 -Action Start
```

入口：

```text
http://localhost:3210/
http://127.0.0.1:8010/api/health
```

默认配置为 `configs/market_v14_local.yaml`，会话目录为 `.local-state-v14-release`。配置名称是兼容身份，不表示没有新版功能。v20.1 从“博弈策略实验室”选择。仓库内脚本使用相对路径，不需要旧外层启动文件。

不要迁移旧 `.venv`、`node_modules` 或 `processes.json`。密钥根据示例自行安全配置，先做离线验证。以上为迁移说明，尚未在新电脑执行。

旧电脑最后一次服务重启被自动审批拒绝，最终说明文字与报告入口未确认被运行进程加载；新电脑重新构建启动后需要核验。

## 6. 实验结果

| 阶段 | 主要实验 | 结论 |
|---|---|---|
| v18 | 2376 局、47520 轮合作竞争评测 | 合成市场验证，不等于现实校准 |
| v19 | 720 源状态、2160 次建议、1080 局合作制度实验 | 预测门禁通过仍可能实际亏损 |
| v20 | 49 例误差隔离、180 校准、180 留出、60 局滚动 | 资产压力筛选未通过同覆盖率比较 |
| v20.1 | 180 新校准、300 新留出、60 局滚动 | 利润有方向性改善，但严格门槛未通过；福利不足 |

v19 按真实经济动作差审计：利润269次干预9负例；福利90次干预19负例。

误差隔离覆盖全部28负例和21严格匹配正例，每例8条未来路径，固定候选。A为公开重建状态/当前策略模型，B为真实状态/当前模型，C为公开状态/原生实际策略及记忆，D为真实状态/实际策略及记忆。相对D的平均绝对误差依次为12347.15、4602.15、11043.17、0合成元。D用原seed精确复现旧记录。只修现金或产能不能解释全部偏差。隐藏真实状态仅用于离线诊断，不能泄漏到公开信息建议器。

v20 同覆盖率利润比较：各18次干预，旧排序0负例，新筛选1负例。不能把降低覆盖率当作质量提升。

v20.1 留出：

- 利润150例中83次干预、1负例，覆盖55.33%，干预负例率1.20%。全案例平均增益9398.92合成元。43个有干预种子簇中1簇亏损，单侧95%簇级风险上界10.5634%，高于预设10%。
- 福利150例中5次干预、1负例，覆盖3.33%，干预负例率20%，均值区间下界不为正。
- 滚动利润单次建议平均+4574.07、每轮建议+17000.07合成元，但分别出现2、4个负局，不能证明长期稳定获益。

两轮独立回放合计960份记录、10560结算步骤、428159算术检查通过。滚动A每轮新原计划无建议，B仅首轮建议，C每轮建议；对手记忆随各自分支演进。

工程覆盖530项Python测试，但并非一次完整530全绿：完整513项中507通过，6项日期依赖随后定向修复复测，另有新增专项。前端19项、构建、类型检查和lint通过。新电脑应重新完整回归。

9次串行H3/H5实测：2公司1.19–1.21秒，5公司2.12–2.21秒，10公司3.79–6.68秒。没有p99或全规模5秒以内的保证。

## 7. 数据与复现

```text
runs/reliable-advisor-v19/
runs/reliable-advisor-v20/
runs/reliable-advisor-v201/
releases/reliable-advisor-v20.1.0/
```

| 版本 | 校准种子 | 留出种子 | 滚动种子 |
|---|---|---|---|
| v20 | 610001–610030 | 620001–620030 | 630001–630010 |
| v20.1 | 640001–640030 | 650001–650050 | 660001–660010 |

看过的留出集不能再作为新方法的独立确认集。`pre-policy-seed-fix` 是保留的错误批次，应按勘误排除。不要删除负例或覆盖冻结发布包。复验先读脚本，输出写入新目录，不盲目运行冻结脚本。

v20.1 冻结 source.zip SHA-256：

```text
c1433e48b091939b3fe79a3deb77f3a94475f2ce13aefeae839aba18f10e6433
```

evidence.zip SHA-256：

```text
7263b354b6aea8a7f90f00367e168a194cd21589d1ce9af75232062119a77bd5
```

## 8. 费用与门禁

最近核验的账本状态：

- 授权累计上限：30元。
- 保守占用：9.993565元；保守剩余：20.006435元。
- 估算累计费用：1.273851元，不是供应商已核对账单。
- 估算构成为导入历史1.093593元，加后续usage估算0.180258元。
- 账本186条历史调用；v19/v20/v20.1近期实验新增付费调用0。
- Codex/ChatGPT自身费用不在此账本。

价格快照于2026-09-08到期，目前ready=False。需实际核价后才能更新，不能仅修改日期绕过保护。经济验收前置门槛仍未满足，不能因为有余额就直接开始真实阶段。

不得重新初始化历史账本、任意释放失败/未知调用预留或超过30元授权。

## 9. 剩余工作

先恢复环境、核验账本与未提交代码，做完整回归、UI保存恢复。用户要求继续修复时，再依次进行：

1. 改善福利建议覆盖率与收益可靠性。
2. 处理利润尾部损失和严格风险上界。
3. 验证多轮累积干预、不同对手策略和规模适应性。
4. 新方法先预注册和冻结，再用新留出验证；保留简单基线、同覆盖率比较和负结果。不得根据测试结果改门槛，或忽略反复补样本的可选停止偏差。
5. 满足前置门槛后，在剩余累计预算内执行真实模型实验。

不要从头重建已有市场、Agent原计划接口或条件响应模型。现实参数校准、真实LLM有效性、全局最优和一般多人纳什均衡仍未证明。

## 10. 新对话开场提示

> 请先阅读交接文档、AGENTS.md、研究日志和最新实验报告，核验新电脑目录、未提交代码、费用账本和实验记录。不要处理PDF，不要把v20.1说成已通过经济可靠性验收。先恢复本机环境并汇报实际核验结果，再执行我指定的后续任务。路径按新电脑适配；真实模型累计预算上限30元，遵守价格及经济验收门禁。
