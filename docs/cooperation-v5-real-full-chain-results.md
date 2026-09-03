# Cooperation v5 真实闭环实验

日期：2026-09-02
证据等级：小样本、方向性真实模型证据；工程回放为确定性验收。

## 1. 目标

上一阶段只在冻结 Observation 中测试了 Agent 是否接受提议，不能证明接受会生成 Commitment，也不能证明目标轮真实动作会履约。本阶段发布独立 `market-v5-cooperation` 配置，并运行最小生产链路：

```text
第1轮：规则 Agent A 私下提议 B 在第2轮贡献100万
  ↓
第2轮：真实 LLM B 接受或拒绝
  ↓
Communication Close 生成非约束 Commitment
  ↓
真实 LLM B 独立提交实际贡献
  ↓
Market 扣除真实成本并更新共享韧性
  ↓
Controller 比较 promised / actual
  ↓
Fulfilled / Partial / Betrayed 与 Credibility 更新
```

每个 Episode 实际配置为 5 轮但只执行前 2 轮，避免目标轮成为终局而被市场正确禁止无未来收益的合作投入。A、C、D 与 B 的第 1 轮全部由 Rule/Mock 完成；真实模型只承担第 2 轮通信和决策，因此 2 个模型 × 3 个共同协议 Seed 共 12 次真实生成任务。

## 2. v5 配置与不可变性

`market_v4.yaml` 保持原 Hash：

```text
sha256:d3139e6939023a81c430e5ab257d2a9808a57c07d1f01d14ddc922bb7b9d2f01
```

新增 `configs/market_v5_cooperation.yaml`：

```text
config_id: market-v5-cooperation
config_version: market-v5.0.0
config_sha256: sha256:bc5ba80831795e29c483bb7a4315da74759e322ebade4a4768c9b82edcbbba61
persona_utilities.capabilities.cooperation: true
```

v5 与 v4 使用相同 `market-env-v4.2.0`，市场公式、人格效用权重和人格 traits 均不变。唯一实质差异是正式允许人格规划层提交合作动作。这样关闭了 G17，又不破坏历史 v4 Advice Hash Replay。

## 3. 零 Token 验收

付费调用前完成：

- v4 Hash 固定测试；
- v4/v5 环境版本、人格权重和 traits 一致性测试；
- v5 完整五轮 Rule/Mock 合作验收；
- Proposal、Acceptance、Commitment、Actual Contribution、Partial Betrayal、Credibility Memory 全部重建；
- Economic、Interaction、Information、Cooperation 四层 Replay 100%；
- 全量 293 项后端测试通过。

## 4. 第一次真实运行：发现“回退伪装成背叛”

第一次 6 个 Episode 中，DeepSeek 结果为 2 次完全履约、1 次部分履约；豆包表面上 3 次都接受但贡献为 0，账本因此正确记录为 3 次背叛。逐层核对 Provider Boundary、Agent Trace 和最终 Joint Action 后发现：豆包原始结构化动作三次都请求贡献 100 万，并明确说要履约；但它把上一轮合作提议的历史 `message_id` 写入本轮 `message_responses`，运行时按可见性规则判为 `INVALID_MODEL_OUTPUT`，随后安全回退动作贡献 0。

因此，这 3 次不是模型主动背叛，而是：

```text
模型请求全额履约
  ↓
历史消息引用不合法
  ↓
整个 Decision fail closed
  ↓
安全回退贡献0
  ↓
合作账本按最终真实动作记录背叛
```

这证明市场与账本执行正确，也证明只看最终合作标签会错误归因模型行为。实验分析必须同时检查 Provider 请求动作、Agent 验证状态、Resolution Source 和最终结算动作。

同一轮还发现 DeepSeek 在自由文本中说“调整为80万”，但结构化 `cooperation_response` 为 `accept`。按协议，accept 正确生成原提议 100 万的 Commitment，所以 80 万实际贡献被判为部分履约。该结果不是账本错误，而是自然语言与结构化协议冲突。

## 5. 协议修复

没有放松消息可见性或自动删除非法字段，而是升级提示协议：

- `market-planner-prompt-v1.18.0` 明确 `message_responses` 只能引用 `current_view.visible_messages`，不能引用 `recent_views`，也不能把 `proposal_id` 当 `message_id`；若当前消息为空必须输出空数组；合作接受已在 Communication 阶段完成，Decision 不再回应历史提议。
- `market-communication-prompt-v1.7.0` 明确结构化 `accept` 是无条件接受原提议完整金额；自由文本不能改金额。若不接受原金额必须 `reject`，当前 MVP 不支持反提议。

定向测试确认新约束进入提示，原有严格运行时校验继续保留。

## 6. 修正后的正式结果

相同 2 个 Provider、相同 3 个共同 Seed、相同 Balanced Persona、相同中性数值合作提示重新运行：

| Provider | Episode | 接受 | 完全履约 | 部分履约 | 背叛 | 平均贡献 |
|---|---:|---:|---:|---:|---:|---:|
| 豆包 | 3 | 3 | 3 | 0 | 0 | 1,000,000 分 |
| DeepSeek | 3 | 3 | 3 | 0 | 0 | 1,000,000 分 |
| 合计 | 6 | 6 | 6 | 0 | 0 | 1,000,000 分 |

12/12 真实生成边界成功，Schema Repair 为 0。每个 Episode 的 Economic、Interaction、Information、Cooperation Replay 和 JSONL Round Trip 全部通过。B 的可信度均从无历史的 500,000 ppm 更新为 750,000 ppm，证明真实履约结果进入了重复博弈信誉状态。

正式运行输入 155,207 Token、输出 4,598 Token，合计 159,805 Token，保守估算 0.331436 元。第一次缺陷诊断运行输入 154,594、输出 5,180，合计 159,774 Token，保守估算 0.333470 元。两轮合计 319,579 Token，保守估算 0.664906 元；创建 Episode 的首次 422 发生在 Provider 调用前，Token 和费用为 0。

## 7. 可以和不可以得出的结论

可以得出：

1. 正式 v5 已能让真实模型走完 Proposal→Acceptance→Commitment→Actual Contribution→Verification→Credibility 的生产链路。
2. Commitment 本身没有改变市场；只有最终实际贡献扣款并形成公共韧性。
3. 严格消息校验能够阻止越权历史引用，但实验必须把安全回退与自主背叛区分开。
4. 在当前 Balanced Persona、充足现金、还有未来轮次、100 万提议的条件下，豆包和 DeepSeek 都稳定接受并履约。

不可以得出：

1. 不能声称 Agent 不会背叛；本样本没有形成背叛行为的条件差异。
2. 不能把 6/6 履约解释为合作权重驱动，因为 v5 没有改变合作效用权重，Persona 仍包含长期性、互惠和承诺诚实 traits，市场也有未来公共收益。
3. 不能声称合作行为“自我演化”；系统仍未做跨代训练或策略种群更新。
4. 不能用本阶段替代此前终局因果实验。此前已经证明豆包可在中性提示下根据纯收益自行选择 0，而 DeepSeek 的零贡献更依赖显式选项框架。

## 8. 下一步

下一步应做最小原生市场背离诱因矩阵，而不是增加合作类型：冻结提议额、公共状态和共同 Seed，只改变 Persona/现金压力或剩余轮数，使履约成本与未来公共收益的权衡真实变化。仍需同时记录 requested action、validation status、settled action 和 verification，任何 fallback 样本从“自主背叛率”中单列。建议先做 2 个模型 × 2 个经济条件 × 3 次重复，共 12 个 Episode、24 次真实生成任务；若零 Token Rollout 没有先证明两条件的最优贡献方向不同，则不得启动付费实验。
