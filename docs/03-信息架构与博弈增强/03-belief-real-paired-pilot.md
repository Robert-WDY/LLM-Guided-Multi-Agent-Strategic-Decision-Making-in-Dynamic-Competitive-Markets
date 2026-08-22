# Belief-aware Planner 真实模型配对 Pilot

## 研究问题

本实验验证两个不同问题：

1. 真实 LLM 是否能读取并明确使用 Controller 生成的对手概率？
2. Belief OFF/ON 是否会改变动作和长期市场路径？

这仍是 Pilot，不是最终因果结论。模型服务即使 `temperature=0` 也未提供可审计的生成 Seed，因此真实调用仍可能有采样或服务端非确定性。

## 实验设计

- 模型：`doubao-seed-2-0-lite-260215`
- Persona：`balanced_v1`
- Information：`public`
- Communication / Cooperation：`off`
- 采样参数：temperature `0`，top_p `1`

### 固定状态反事实

在 Seed 700 的同一 round-4 State 中，预先制造三轮公开价格证据，使 company_B 下一轮降价概率达到 `666667 ppm`。除 Belief treatment 和由其决定的 Observation Hash 外，其余 State、约束、Persona、模型和 Prompt 版本完全相同。

条件：

- `belief_off`：Belief State 为 null；
- `belief_on`：包含完整公司级公开动作概率。

每个条件真实调用 3 次，顺序交替，共 6 次调用。

### 多轮共同 Seed

共同 Seeds：701、702、703。每个 Seed 均运行：

- Belief OFF；
- Belief ON。

两种条件使用相同 Episode ID、Seed、市场配置、1 LLM + 3 Rule 构成和 Persona。每个 Episode 5 轮，共 6 个 Episode、30 次真实模型决策。Seed 702 反转 ON/OFF 运行顺序，降低固定顺序影响。

## 工程结果

- 36/36 次真实调用成功；
- 6/6 Episode 完成；
- LLM fallback 为 0；
- Economic Replay 100%；
- Interaction Replay 100%；
- Information Replay 100%；
- Belief Replay 100%；
- RoundEvent 日志可在进程退出后重建 Manifest/MarketTransition 并恢复 Summary。

总 Token：

| 阶段 | 调用 | Input | Output | Total |
|---|---:|---:|---:|---:|
| 固定状态 | 6 | 38,133 | 3,345 | 41,478 |
| 多 Seed Episode | 30 | 343,160 | 18,801 | 361,961 |
| 合计 | 36 | 381,293 | 22,146 | 403,439 |

多轮条件中：OFF 为 177,104 Tokens，ON 为 184,857 Tokens。Belief ON 增加 8,488 输入 Tokens、减少 735 输出 Tokens，净增加 7,753 Tokens，约为 OFF 的 4.38%。

## 固定状态结果

修正关键词口径后：

- OFF 明确信念/概率引用率：0/3；
- ON 明确信念/概率引用率：3/3；
- 3/3 配对的 Requested Action 均不同；
- ON 相对 OFF 的价格差：`-100 / -50 / -50` 分；
- 平均价格差：`-66.67` 分，三次方向一致；
- ON 平均广告预算增加 100,000 分；
- ON 平均服务预算增加 66,667 分；
- ON 平均韧性预算减少 26,667 分。

这提供了比“Prompt 中出现 Belief”更强的管线证据：真实模型不仅在文字中明确提到对手降价概率，而且在三个重复中均把报价向下调整。但样本只有 3 对，且预算响应不完全一致，不能据此估计稳定效应大小。

## 多轮结果

### 信念是否被使用

OFF 的明确概率/信念引用率为 0%；ON 为：

- Seed 701：80%；
- Seed 702：100%；
- Seed 703：100%。

平均 ON-OFF 引用率差为 93.33 个百分点。Seed 701 的第一轮只有中性先验，模型未明确引用，随后有公开历史后开始引用，符合处理语义。

### 动作和市场路径

三个 Seed 的 ON/OFF 最终 State Hash 均不同，说明真实模型动作差异足以让长期市场路径分叉。

ON-OFF 的平均 Requested Action 差异：

- 价格：`+84.8` 分；
- 广告预算：`-126,667` 分；
- 服务预算：`-46,667` 分；
- 私人韧性预算：`-93,333` 分；
- 产能与共享韧性贡献：无变化。

这与固定高降价信念下的短期降价并不矛盾：多轮中概率和市场状态不断变化，且一旦首轮动作不同，后续 State 本身就会分叉。因此长期均值不是单一 Prompt 字段的直接效应。

### 利润

市场总利润 ON-OFF：

- Seed 701：`+12,781,823` 分；
- Seed 702：`+1,099,339` 分；
- Seed 703：`-3,091,517` 分。

均值为 `+3,596,548` 分，但只有 2/3 Seeds 为正，且均值被 Seed 701 强烈影响。LLM 公司自身利润差也为 2 正 1 负，均值 `+850,777` 分。当前只能说“收益方向异质”，不能说 Belief 提高利润。

## 信念预测质量

三个 ON Episode 共 45 个 observer→opponent 预测：

- 平均 Top-1 Accuracy：44.44%；
- Mean Brier Score：0.6631；
- Mean Log Loss：1.0694。

三分类均匀基线的 Brier 约为 0.6667、Log Loss 约为 1.0986。因此本次简单累计频率信念只有很弱的信息增益，远低于前一确定性稳定 Mock 路径的 80% Accuracy。它已经足以被模型读取并改变动作，但预测器本身还不够强。

## 实施中发现的问题

### 1. Episode 汇总器误读 companies 结构

第一组 5 轮已经完整结算并落盘，但汇总器把 `state_after.companies` 当成 list；实际 RoundEvent 使用 company-id mapping，导致写 Summary 前异常。

修复：兼容 mapping/list，并新增从完整 RoundEvent JSONL 重建 Manifest 和 MarketTransition 的恢复路径。Seed 701 OFF 的 5 次调用没有重复付费，恢复后四层 Replay 全部通过。

### 2. 关键词指标产生假阳性

初版把“降价/持平/涨价”都算作 Belief 引用，导致 OFF 组也接近 100%，因为普通竞争分析本来就会使用这些词。

修复：拆分为普通方向措辞与“对手概率/预测/信念/公开价格历史”明确引用。使用保存的 Planner Output 重新分类，不重新调用模型。修正后固定状态 OFF=0%、ON=100%。

### 3. 真实模型非确定性仍存在

相同固定状态下，同一条件的价格和预算仍有一定离散。固定状态设计、重复调用和顺序交替降低了混杂，但 3 对样本不足以完全分离 Belief 效应与服务端随机性。

## 结论与下一步

当前已经有初步证据支持：

> 真实 LLM 能识别结构化公开动作信念，并在固定状态下以方向一致的方式调整价格。

但当前更新器的预测增益很弱，利润方向不稳定。下一步不应立即加入更多隐藏类型，而应：

1. 将固定状态重复扩到至少 10 对；
2. 增加 `neutral / high-cut / high-maintain / high-raise` 四种概率处理；
3. 加入完全相同 treatment 的重复 placebo，估计模型自身动作方差；
4. 多轮扩到至少 10 个共同 Seed；
5. 比较累计频率、滑动窗口和带遗忘因子的更新器，再决定是否进入对手类型信念。

## 产物

- `runs/belief-real-paired-20260821/summary.json`；
- `runs/belief-real-paired-20260821/fixed-state/calls.json`；
- `runs/belief-real-paired-20260821/fixed-state/summary.json`；
- `runs/belief-real-paired-20260821/seed-*/belief_*/round-events.jsonl`。
