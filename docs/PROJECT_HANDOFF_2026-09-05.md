# Game Theory Agent 项目交接文档

交接日期：2026-09-05  
工作目录：`C:\Users\12204\Desktop\game-theory-agent`  
当前分支：`feat/final-strategic-market-v6`  
当前 HEAD：`708d532`（与本地 `main`、`origin/main` 相同）  
当前研究版：`market-env-v10.0.0` / `market-v10-multi-objective`

## 1. 交接结论

项目目前已经形成一个可运行、可回放、可做多 Agent 实验的动态市场研究平台。系统从最初的单轮企业决策扩展到：消费者选择、公司财务与破产、不完全信息、Belief、Opponent Model、Persona数学规划、长期Market Rollout、通信与合作、搭便车、阈值公共项目、应急互助、价格协调与监管、上游供应链、消费者/上下游/政府福利账本，以及多目标企业。

当前最重要的边界是：工程和合成机制验证较完整，但现实外部有效性尚未完成。消费者支付意愿、供应商参数、外部成本和多目标权重仍是显式合成先验；真实LLM实验能提供方向性证据，但尚不足以支持跨模型或现实行业统计结论。

## 2. 接手前必须知道的工作区状态

当前大量最终市场代码位于工作树中，尚未提交到当前 HEAD。不要执行`git reset --hard`、`git checkout -- .`或未经审查的清理命令。接手后的第一步应是阅读`git status --short`和`git diff`，再按功能分组提交。

特别注意：

- `PROJECT_EVOLUTION_RECORD.md`是本地研究总记录，被`.git/info/exclude`排除，禁止上传GitHub；
- 它必须与`C:\Users\12204\Desktop\洞墟科技\项目\信息总结\错误总结\PROJECT_EVOLUTION_RECORD.md`保持完全一致；
- 当前两份记录的SHA-256为`8CAA0562F8904D5CDC81A28E20B1FF382AAF1F1087EED38E1619C927D4062739`；
- `runs/`是本地实验输出并由`.gitignore`排除；
- 根目录的`ARIN7600_Interim Report.pdf`和`game-theory-agent.zip`是用户文件，不要误删或自动加入提交；
- 不要提交`.env`、API Key、Controller Token、原始凭据或运行日志。

建议提交时使用显式路径`git add`，不要直接`git add .`。

## 3. 最终版本启动方式

### 3.1 安装

要求Python 3.11+和Node.js 22.13+。

```powershell
cd C:\Users\12204\Desktop\game-theory-agent
python -m pip install -e ".[test]"
npm --prefix frontend ci
```

### 3.2 本地演示启动

分别打开两个PowerShell窗口：

```powershell
cd C:\Users\12204\Desktop\game-theory-agent
.\scripts\start_final_backend.ps1
```

```powershell
cd C:\Users\12204\Desktop\game-theory-agent
.\scripts\start_final_frontend.ps1
```

入口：

- 前端：`http://127.0.0.1:3210`
- 私有市场API：`http://127.0.0.1:8010`
- Agent Gateway：`http://127.0.0.1:8011`
- 健康检查：`http://127.0.0.1:8010/api/health`

启动脚本只在回环地址启用本地无令牌浏览器入口。非本机部署必须关闭`MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER`并设置高熵`MARKET_CONTROLLER_TOKEN`。

### 3.3 真实模型

`.env.example`支持豆包、DeepSeek和Mock。真实调用必须显式设置本地API Key，并在实验命令中加入`--authorize-real-model`。不要把Key写入配置、预注册、结果或Git。

## 4. 当前系统架构

```text
版本化市场配置
    ↓
MarketEnv：消费者、公司、事件、供应链、政府与福利结算
    ↓
ObservationBuilder：按公司生成可见信息与观察Hash
    ↓
Belief / Opponent Model / Utility Inference
    ↓
Persona Utility + Authoritative Market Rollout + Pareto/Reliability Gate
    ↓
已发布建议（弃权时不进入Agent上下文）
    ↓
LLM / Rule / Human Action
    ↓
统一动作验证与决策解析
    ↓
同步联合结算、事件日志、Memory、Replay、Evaluator
```

关键源码：

- `src/game_theory_agent/market/environment.py`：权威市场结算；
- `src/game_theory_agent/market/models.py`：状态和动作契约；
- `src/game_theory_agent/market/strategic.py`：破产、集中度、价格战和最终战略状态；
- `src/game_theory_agent/market/supply_chain.py`：供应商、采购分配和中断传导；
- `src/game_theory_agent/market/welfare.py`：正式经济福利账本；
- `src/game_theory_agent/agents/observation.py`：可见性过滤和观察构建；
- `src/game_theory_agent/agents/personas.py`：Persona档案、效用权重和评估；
- `src/game_theory_agent/agents/prompt_builder.py`：中文决策上下文；
- `src/game_theory_agent/strategic_reliability/rollout.py`：真实MarketEnv候选动作模拟；
- `src/game_theory_agent/strategic_reliability/reliable_planner.py`：长期建议可靠性门禁；
- `src/game_theory_agent/advisor/context_view.py`：建议进入Agent上下文的隔离边界；
- `src/game_theory_agent/decisioning.py`：请求动作到可执行动作的统一解析；
- `src/game_theory_agent/orchestration/coordinator.py`：多Agent轮次编排；
- `src/game_theory_agent/api.py`：Controller API与Agent Gateway；
- `frontend/app/page.tsx`：研究控制台主页面。

## 5. 市场版本关系

| 配置 | 主要新增能力 | 状态 |
|---|---|---|
| `market_v6_final.yaml` | 破产、垄断、价格战、公共韧性、阈值项目、应急互助、价格协调与监管 | 历史证据冻结 |
| `market_v7_consumer_calibrated.yaml` | 显式WTP群体、硬支付意愿门槛、不购买选择 | 合成形状验证通过 |
| `market_v8_supply_chain.yaml` | 上游供应商、采购拥堵、中断、备选与分散采购 | 合成机制验证通过 |
| `market_v9_welfare.yaml` | 消费者、上下游、政府和外部性分账 | 会计与方向验证通过 |
| `market_v10_multi_objective.yaml` | 社会福利目标与三类多目标企业 | 当前默认最终入口 |

新版本通过受限`extends`继承旧配置。不要原地修改已有证据对应配置；新参数应创建新版本和新Hash。

## 6. 当前已完成的功能

### 6.1 市场与消费者

- 多家公司同步行动，份额总量守恒；
- 价格、广告、服务、品牌、可靠性和缺货共同影响消费者选择；
- 五档显式支付意愿群体；价格超过WTP时不能购买；
- Outside Option、自愿不购买和缺货后流失分开记录；
- 事故、市场事件、服务冲击、现金、利润、企业价值与破产退出；
- 活跃公司、HHI、主导企业、垄断、价格战和监管压力。

### 6.2 信息与Agent

- 完全信息和公开不完全信息；
- PublicState、PrivateState、ObservationEnvelope和company-scoped observation hash；
- Belief、Opponent Model、Utility Inference与动作预测；
- 对手现金、成本、利润、Persona和私人事故隔离；
- 长短期Memory、合作信誉、决策Trace和Replay；
- LLM、Mock、Rule和Human席位。

### 6.3 合作与竞争

- Shared Resilience Contribution及公共品搭便车；
- Proposal、Acceptance、Commitment、Actual Contribution、履约/部分履约/背叛、Credibility；
- 阈值型冷链公共项目，表达猎鹿/协调失败；
- 双边应急互助，需要双方同轮匹配且受真实闲置产能约束；
- 价格协调、单方低价背叛、监管调查和罚款；
- 价格协调候选仅限研究，不允许Advisor作为生产建议发布。

### 6.4 供应链与福利

- 经济型和韧性型供应商；
- 主供应商、备选供应商和主采购份额；
- 有限上游产能、同比例分配、采购成本、供应商剩余与中断；
- 消费者剩余、下游生产者剩余、上游生产者剩余、政府罚款收入、执法成本、缺货和退出外部成本；
- 罚款按主体间转移核算，执法成本才是资源损失。

### 6.5 多目标企业

旧Persona继续支持利润、份额、增长、现金、风险、信誉、韧性与合作。v10新增：

- `stakeholder_balanced`：企业价值和利益相关者折中；
- `public_service`：可负担价格、服务连续和消费者福利；
- `resilience_steward`：长期韧性与供应连续。

社会福利权重进入数学效用与真实MarketEnv Rollout，不只是提示词。

## 7. 已验证的重要研究结果

### 7.1 真实LLM证据

- Strategic Advisor核心对照：35个完整配对中22正、8平、5负，平均企业价值+381,344分，平均候选遗憾降低311,310分；
- 5个负样本全部指向“弃权但候选仍进入上下文”的锚定问题；彻底隔离后，弃权与无Advisor观察Hash相同；
- 对手模型最小实验：正确画像比打乱画像平均+421,788分，2/3人格为正；
- 1 LLM+3 Rule长局只有2个完整配对且一正一负；v9释放建议数为0，因此不能解释为Advisor长期效果；
- LLM消费者18次调用能区分价格、品牌和服务偏好，但高价Outside选择率为0，不能代替真人数据。

紧邻阶段真实实验已知至少使用2,088,849 Token，保守估算6.717867元；其中一个失败调用无法恢复完整Usage，所以这是下限。

### 7.2 确定性合成市场证据

- 消费者v7：价格8,000→18,000分时焦点需求4,741→46；20/20 Seed方向正确；
- 供应链v8：强制低价供应商中断时，分散采购把平均缺货约6,989降到105；
- 福利v9：低价提高消费者剩余但压缩企业利润；高价使大量消费者退出并显著降低总福利；
- 多目标v10：公共服务人格20/20 Seed选择可负担价格，纯企业利益基线20/20选择均衡价格；
- 供应冲击下，集中采购的焦点利润-907,025、缺货6,990、总福利-5,522,675；分散采购利润3,308,950、缺货124、总福利63,053,032。

完整数字和证据边界见`docs/stage11-consumer-supply-welfare-multi-objective-v10.md`。

## 8. 重要实验产物

不要只看Markdown结论，应优先读取JSON中的门禁、行级数据和Hash。

| 主题 | 规范/预注册 | 结果 |
|---|---|---|
| Advisor真实核心 | `experiment-specs/final-market-real-core-v1/` | `runs/final-market-real-core-v1/summary.json` |
| Advisor弃权修复 | `experiment-specs/final-market-abstention-repair-real-v1/` | `runs/final-market-abstention-repair-real-v1/summary.json` |
| 对手模型价值 | `experiment-specs/final-market-opponent-model-value-real-v1/` | `runs/final-market-opponent-model-value-real-v1/summary.json` |
| 多轮真实市场 | `experiment-specs/final-market-multiround-real-v1/` | `runs/final-market-multiround-real-v1/summary.json` |
| LLM消费者 | `experiment-specs/llm-consumer-pilot-v1/` | `runs/llm-consumer-pilot-v1/summary.json` |
| 消费者形状 | 无Provider | `runs/consumer-market-calibration-v1/summary.json` |
| 供应链 | 无Provider | `runs/supply-chain-v8-validation/summary.json` |
| 福利核算 | 无Provider | `runs/welfare-v9-validation/summary.json` |
| 多目标企业 | 无Provider | `runs/multi-objective-v10-validation/summary.json` |
| 最终发布门禁 | 读取既有证据 | `experiment-results/stage7-final-release-v1/summary.json` |

关键结果Hash：

- Advisor核心：`sha256:4dd0a83a1abdab96359196d67266ed189a1bd8bcf3e81f74304ee9d9a70efb59`
- 弃权修复：`sha256:1a2699e2bc1fe65be94adf711aadfc6ad2a2a6e443ec1f79b88f300ff801efc0`
- 对手模型：`sha256:e5d9ae56a6fb518145700128526edcc1b3541af8ede2735e2033b69bc46cf830`
- 多轮真实：`sha256:9528f4481a486713dc88afc5d5d334c4abc387fe82b7ac9f31c5b333b7b15448`
- LLM消费者：`sha256:a2097fbee14ec781f6b379438a4677bac658fa57df21788e4f6206bede7f669c`
- 消费者v7：`sha256:c6c64e19efc4354a6bfde19b5edfc2a51c936023fb7f911c4c166856632e82b2`
- 供应链v8：`sha256:ab35403b6e70666d07a0756e8e9300d29d82f034343cf2626299aafe221d291f`
- 福利v9：`sha256:74b2e35ba82bf0d791939cb1e1dd47ca409ab2b89465eb89cad02b7b8250f7fd`
- 多目标v10：`sha256:c3a50afb502715431ee10b8aea0edbb983454918a47f7d8c4f0f7eb322574da6`

## 9. 验证与复现实验

### 9.1 工程回归

```powershell
cd C:\Users\12204\Desktop\game-theory-agent
$env:PYTHONPATH="src"
python -m pytest -q

npm --prefix frontend run lint
npm --prefix frontend test
```

交接时的结果：后端337项测试全部通过；前端生产构建和5项测试通过。仅有第三方Starlette multipart弃用警告，不影响结果。

### 9.2 零Token实验

```powershell
$env:PYTHONPATH="src"
python -m game_theory_agent.experiments.consumer_market_calibration_v1
python -m game_theory_agent.experiments.supply_chain_v8_validation
python -m game_theory_agent.experiments.welfare_v9_validation
python -m game_theory_agent.experiments.multi_objective_v10_validation
```

### 9.3 真实模型实验

先运行`--prepare`检查计划和预算，再显式授权。不要为了验证文档或Hash重复付费运行。

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="本地密钥"
python -m game_theory_agent.experiments.llm_consumer_pilot_v1 --prepare
python -m game_theory_agent.experiments.llm_consumer_pilot_v1 --authorize-real-model
```

其他真实实验采用相同两阶段接口。任何供应商失败、解析失败或落盘失败都必须保留Incident并计入调用预算，不得静默补样。

## 10. 已知缺口与风险

### P0：进入下一轮研究前处理

1. **现实参数未校准。** 当前WTP、弹性目标、供应商成本/可靠性、缺货和退出外部成本均为合成参数。不能宣称对应现实行业。
2. **Advisor长局处理稀疏。** 最近20个处理轮次释放建议为0；下一次真实长局前必须先找到满足发布门禁的冻结状态，并设置最低释放率门禁。
3. **前端尚未完整展示v10。** 前端能识别v10为最终市场，但还没有完整的供应商选择器、采购结果、消费者剩余、上下游剩余和政府账本面板。
4. **Persona能力接口仍主要暴露中期人格。** `/api/v1/capabilities/personas`使用固定的中期人格列表；v10新人格可通过API请求运行，但前端配置入口尚未全部暴露。

### P1：下一阶段功能

1. 供应商目前不是自主Agent，不能主动定价、扩产、签长期合同或议价；
2. 政府目前是固定执法规则和账本主体，不会选择执法强度、罚款、韧性补贴或消费者补贴；
3. 多目标v10仅完成确定性数学层验证，尚未完成真实LLM采纳和10—20轮目标一致性验证；
4. 分散采购在当前参数下表现强，需要在价差、容量、共同中断相关性和合同成本Holdout中检验；
5. 福利账本没有劳动者工资/就业、环境排放、税收扭曲和供应商破产等主体；不要一次全加，应逐机制验证。

### P2：长期研究

1. 用真实购买或正式问卷校准消费者分布；
2. 形成供应商—企业—消费者—政府的层级/动态博弈；
3. 将多目标、供应链和政府动作纳入Self-play与策略人口；
4. 扩大跨模型、跨Seed和未见对手实验；
5. 满足数据量、来源、特征一致性和Holdout门禁后，再训练策略选择器或微调小模型。

## 11. 推荐接手顺序

### 第一步：冻结和提交当前v10

先审查工作树并按以下逻辑分组提交：

1. v6最终战略市场与Advisor/Self-play；
2. v7消费者WTP；
3. v8供应链；
4. v9福利账本；
5. v10多目标企业；
6. 前端、启动脚本、测试与文档。

提交前重新运行337项后端测试和前端测试。不要把`runs/`、研究总记录、PDF、ZIP或密钥加入提交。

### 第二步：补齐v10前端与能力接口

让用户在配置页选择多目标人格和采购策略；实时页显示供应商中断、采购履约、缺货传导、消费者剩余、上下游剩余、政府净预算与总福利。所有解释使用中文，并区分本轮值和累计值。

### 第三步：做多目标真实LLM最小实验

固定3个Seed、3种人格、2个冻结状态，比较无Planner建议与v10多目标建议。先用零Token确认建议会实际发布，再运行不超过36次真实调用。主要看动作方向、采纳率、企业价值、消费者剩余、总福利和最坏结果，不只看平均利润。

### 第四步：供应商战略化

一次只加入一种上游动作。建议先做供应商报价与产能投资，不要同时增加合同、纵向并购和转账。用规则Agent证明双重加价、可靠性投资和集中采购风险，再接LLM。

### 第五步：政府政策主体

先把政府动作限制为执法强度和一种韧性补贴。验证罚款转移、执法成本、补贴财政成本、企业响应、消费者剩余和总福利闭合后，再做政府—企业动态博弈。

## 12. 研究结论的表达规范

可以说：

> 系统已经通过多Seed、确定性Replay证明消费者退出、供应中断传导、福利分账和多目标效用在合成市场中的因果链成立；小样本真实LLM实验发现Advisor和对手画像具有方向性下游价值。

不可以说：

> 当前参数代表现实消费者；分散采购普遍最优；政府政策已优化；Advisor已在长局中稳定提高收益；当前人格等于真实企业；LLM消费者等于真人调查。

## 13. 相关文档

- `README.md`：项目入口与总体能力；
- `PROJECT_EVOLUTION_RECORD.md`：完整跨阶段研究记录，本地专用；
- `docs/stage11-consumer-supply-welfare-multi-objective-v10.md`：本阶段详细报告；
- `docs/stage7-final-strategic-market-v6.md`：最终竞争、合作、Advisor、Self-play和迭代市场；
- `docs/persona-research.md`：人格设计与效用；
- `docs/incomplete-information-p0-p5.md`：不完全信息和Belief；
- `docs/frontend-research-dashboard.md`：前端研究控制台；
- `docs/agent-runtime-and-orchestration.md`：Agent运行时与协调器。

## 14. 接手检查清单

- [ ] 已确认工作目录和分支；
- [ ] 已阅读`git status --short`，未覆盖未提交改动；
- [ ] 已确认v10健康检查返回`market-v10-multi-objective`；
- [ ] 已运行后端337项测试；
- [ ] 已运行前端Lint、Build和5项测试；
- [ ] 已阅读Stage 11报告和研究总记录；
- [ ] 已确认两份研究总记录Hash一致；
- [ ] 已理解真实LLM证据与合成证据的区别；
- [ ] 新真实实验先预注册、预算和检查Treatment释放率；
- [ ] 新配置使用新版本，不修改已有证据对应配置；
- [ ] 提交中不含`runs/`、密钥、研究总记录、PDF或ZIP。
