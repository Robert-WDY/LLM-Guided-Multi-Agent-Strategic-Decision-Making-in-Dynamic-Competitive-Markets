# 阶段 2 真实模型小型 Smoke：Seed 810

- 日期：2026-08-20
- 模型：`doubao-seed-2-0-lite-260215`
- 构成：2 LLM（company_A/B）+ 2 Rule（company_C/D）
- 轮数：5
- 人格：A/B 均为 `balanced_v1`
- 对照：相同市场 Seed、模型、人格、采样参数；`off` 对比 `public_private`

## 1. 工程验收

两组都满足：

- 5/5 轮完成，10/10 真实 LLM 经营决策成功；
- LLM fallback 为 0，Rule fallback 数量符合预期；
- 每轮只结算一次，没有部分状态更新；
- 市场 Replay 100%；
- Interaction Replay 100%；
- 落盘 RoundEvent 与内存事件完全一致；
- `protocol_passed=true`、`passed=true`。

通信组额外完成 10/10 次真实通信生成，无重试、无 Schema 错误、无超时。

## 2. 真实通信行为

`public_private` 允许公开消息和私信，但两个 balanced Agent 自主选择：

- 公开消息：10；
- 私信：0；
- speech act：10 条均为 `statement`；
- 决策阶段消息回应：11；
- `accepted`：5；
- `ignored`：6；
- `rejected`：0。

主要话术是维持合理价格、避免恶性价格战、保留现金缓冲、共同维护稳定竞争和行业盈利空间。没有提出结构化的对手动作请求，全部是自身非绑定行动声明。

自身声明共有 50 个结构化字段，其中 37 个与最终动作一致，alignment 为 74%。这说明模型会表达计划，但声明不是合同，仍可能在决策阶段调整。

## 3. 第一轮直接观察

两组第一轮的 Market、Companies、round 和 state_version 完全相同；State Hash 因 Episode ID 不同而不同。

第一轮动作：

| 公司 | off | public_private |
| --- | --- | --- |
| A | 价格 10000；广告 500000；服务 500000；韧性 250000 | 价格 10000；广告 500000；服务 600000；韧性 200000 |
| B | 价格 9800；广告 500000；服务 500000；韧性 300000 | 价格 10000；广告 0；服务 500000；韧性 200000 |

B 在审计理由中明确接受 A 关于稳健经营、反对恶性价格竞争的公开声明。B 同时提高报价并削减广告，说明消息确实进入了判断过程，并伴随动作变化。

但这仍是 Smoke 证据，不是严格因果估计：真实模型即使 temperature=0 也可能存在服务端非确定性，而且两组是两次独立推理调用。正式结论需要多 Seed、公司位置轮换和重复运行。

## 4. 五轮结果差异

| 指标 | off | public_private | 差异 |
| --- | ---: | ---: | ---: |
| 市场累计利润（分） | 23,318,555 | 21,453,717 | -1,864,838（-7.997%） |
| 累计缺货订单 | 7 | 167 | +160 |
| 累计外部选项订单 | 6,773 | 6,415 | -358 |
| 平均价格离散度（分） | 239.581 | 335.007 | +95.426 |
| 平均需求-产能绝对缺口 | 716.85 | 946.65 | +229.80 |

通信组第四轮 A/B 都没有扩产，而 off 组 A/B 各投入 800000 分扩产。这是五轮利润和缺货差异的重要伴随因素，但单 Seed 下不能判定为通信造成的稳定规律。

## 5. 调用开销

- off 决策调用：输入 130886 tokens，输出 6502 tokens，累计延迟 154623 ms；
- public_private 决策调用：输入 155349 tokens，输出 7708 tokens，累计延迟 174571 ms；
- public_private 通信调用：输入 129622 tokens，输出 2043 tokens，累计延迟 63177 ms。

通信历史显著增加输入 Token。下一步应研究压缩最近通信历史和 Prompt 固定字段，避免小型实验也产生过高上下文成本。

## 6. 结论

本次 Smoke 已证明真实模型链路可以：生成消息、正确关闭和过滤视图、把消息送入决策、记录采信或忽略、完成市场结算并通过双 Replay。

它没有证明通信能够提高利润、形成合作或稳定抬价。当前只看到 balanced 人格自然生成了偏向“避免价格战”的公开协调话术，同时伴随更高价格离散、较少扩产和较低五轮利润。正式研究至少需要共同 Seeds、位置轮换、重复调用，并把 Seed/Episode 作为统计单位。

## 7. 原始产物

- `runs/_interaction-real-smoke-seed810-off-20260820/summary.json`
- `runs/_interaction-real-smoke-seed810-off-20260820/round-events.jsonl`
- `runs/_interaction-real-smoke-seed810-public-private-20260820/summary.json`
- `runs/_interaction-real-smoke-seed810-public-private-20260820/round-events.jsonl`
