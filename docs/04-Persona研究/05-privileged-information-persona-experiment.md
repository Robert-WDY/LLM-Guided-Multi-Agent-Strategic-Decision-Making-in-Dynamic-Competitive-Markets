# 单一完全信息 Agent 人格夺冠实验

## 研究问题

在市场整体采用不完全信息的情况下，只让一家公司看到其他公司的完整状态，是否足以让它跨人格稳定获得第一名？

这个问题不能预设答案。工程验收只负责证明“信息确实被正确送达且没有扩散给其他公司”；是否夺冠属于实验结果。

## 实验处理

固定目标公司为 `company_A`，市场级信息模式始终为 `public`：

| 条件 | 目标公司看到什么 | 其他公司看到什么 |
| --- | --- | --- |
| 公开信息对照 | 对手价格、份额、销量、声誉 | 公开信息 |
| 单一完全信息处理 | 对手完整公司状态，包括财务、运营、风险和 Persona | 公开信息 |

完整信息不包括对手尚未提交的本轮行动，也不包括未来随机数。因此它是“当前状态全知”，不是“预知未来”。

公司级处理通过 `observer_information_modes={"company_A": "perfect"}` 写入 Episode Manifest。特权观察必须使用公司令牌读取；Observation Hash、RoundEvent 和 Information Replay 均按公司实际处理模式验证。

## 实验矩阵

- 目标公司：1 个模型 Agent；
- 对手：3 个固定 Rule Agent；
- 人格：均衡经营、极端激进、极端保守、长期自利、短期逐利；
- 条件：公开信息对照、单一完全信息处理；
- 配对单位：人格 × Seed；
- 两个条件复用相同 Seed、相同 Episode ID、相同市场与对手；
- 条件调用顺序交替，降低真实模型供应商的时间顺序影响。

主要名次按最终综合企业价值计算，同时保留总资产、累计利润和市场份额。

## 硬工程验收

- 市场级信息模式仍为 `public`；
- 目标公司处理组每轮均为 `perfect` 观察；
- 目标公司能看到每个对手的财务、运营、风险和 Persona；
- 对照组私有字段泄露数为 0；
- 特权观察不能匿名读取；
- 公共状态对所有公司完全一致；
- 经济、交互、观察、信念和博弈模块 Replay 全部通过。

“目标公司第一名”不是工程通过条件，否则会把负面研究结果错误地隐藏成程序失败。

## 确定性工程运行

```powershell
.\.venv\Scripts\python.exe -m game_theory_agent.experiments.privileged_information_persona `
  --provider mock `
  --seeds 6101,6102,6103 `
  --personas balanced_v1,aggressive_v1_extreme,conservative_v1_extreme,selfish_long_term_v1,profit_myopic `
  --rounds 10 `
  --output runs/privileged-information-persona-mock-v2-20260821
```

共 30 个 Episode、300 个目标 Agent 决策。工程验收全部通过。确定性决策器没有读取新增的对手私有字段，因此两条件的行动、市场路径和最终结果完全相同：企业价值、利润、份额和名次的平均配对差均为 0。两个条件的第一名率均为 33.33%。

这个结果证明：获得信息和利用信息是两件不同的事。仅把完整状态加入上下文，不会自动转化为竞争优势。

第一次确定性试跑曾把条件名写进 Episode ID，导致部分随机组件获得不同命名空间，从第 4 轮开始产生不同随机市场路径。该结果已标记为无效工程试跑。修复后，配对条件复用相同 Episode ID，再次运行得到上述零差异结果。

## 真实模型运行

```powershell
.\.venv\Scripts\python.exe -m game_theory_agent.experiments.privileged_information_persona `
  --provider doubao `
  --seeds 6101 `
  --personas balanced_v1,aggressive_v1_extreme,conservative_v1_extreme,selfish_long_term_v1,profit_myopic `
  --rounds 10 `
  --temperature 0 `
  --top-p 1 `
  --resume `
  --output runs/privileged-information-persona-doubao-3seed-20260821
```

先导实验原计划使用 1 个共同 Seed，覆盖 5 种人格、两个条件，共 10 个 Episode、100 次真实模型决策。实际完成 3 种人格的 6 个有效 Episode、60 次真实模型决策后，豆包账户返回 `403 AccountOverdueError`，长期自利处理组第 9、10 轮进入规则回退，因此该 Episode 被硬验收拒绝并移动到可恢复的失败归档目录；短期逐利尚未运行。

已完成结果：

| 人格 | 公开信息名次 | 完整信息名次 | 企业价值变化 | 累计利润变化 | 份额变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 均衡经营 | 4 | 4 | +8,773,398 分 | +9,074,038 分 | -4.2668 个百分点 |
| 极端激进 | 4 | 4 | -2,836,100 分 | -1,098,985 分 | +0.1507 个百分点 |
| 极端保守 | 4 | 4 | +773,150 分 | -1,367,871 分 | +0.3329 个百分点 |

三个完整信息处理组均未取得第一，已足以否定“只给完整状态即可让所有人格稳定第一”这一强命题。均衡人格明显改善企业价值和利润，但仍未改变名次；激进人格反而受损；保守人格以利润下降换取了小幅综合价值提升。这说明完整信息确实改变了真实模型决策，但不同人格利用信息的方向不同。

6 个有效 Episode 共使用输入 Token 901,241、输出 Token 38,708、总 Token 939,949。被拒绝 Episode 的前 8 次成功调用仍实际消耗输入 121,997、输出 5,381、总计 127,378 Token。因此本次真实运行供应商实际成功调用总量为 68 次，输入 1,023,238、输出 44,089、总 Token 1,067,327；所有成功调用都有供应商 Token 记录。结果见 `runs/privileged-information-persona-doubao-3seed-20260821/partial-research-summary.json`。这是单 Seed 的方向性真实模型证据，不是多 Seed 统计结论。

## 解释边界

即使真实模型处理组不能全部夺冠，也不能据此认为完整信息无价值。可能原因包括：

1. 模型没有有效提取或比较私有字段；
2. Persona 目标与综合企业价值排名不一致；
3. 当前行动仍是同时提交，完整状态无法揭示对手本轮尚未决定的行动；
4. 完整信息增加上下文长度和认知负担；
5. 10 轮市场不足以兑现部分长期信息优势。

若完整信息本身不足以稳定夺冠，下一项处理应是“完整信息 + 明确的状态比较摘要”，再之后才是“完整信息 + 反事实决策器”。不能把后两者的效果归因给原始信息优势。
