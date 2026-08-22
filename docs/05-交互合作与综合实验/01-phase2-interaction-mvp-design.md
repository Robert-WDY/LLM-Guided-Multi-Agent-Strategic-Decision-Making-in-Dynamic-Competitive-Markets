# 阶段 2：Interaction MVP 设计与验收

状态：设计已冻结为 `interaction-v1-draft`；P1/P2/P3 工程实现与 Mock 验收已完成，真实 LLM 小型 Smoke 待执行。

## 1. 阶段定位

阶段 2 是一个可审计的非约束性通信层，也就是博弈论中的 Cheap Talk。它允许 Agent 公开发言、点对点私聊、提出建议、承诺、威胁或撒谎，但消息本身没有合同效力，也不能直接改变市场。

本阶段继续保持：

- `cooperation_enabled = false`；
- 不增加合作人格、社会福利权重、转账、联合投资或共同动作；
- 每个 Agent 最终仍独立提交自己的经营动作；
- `MarketEnv`、`MarketState`、市场公式和动作安全约束不因通信而改变。

本阶段要回答的工程问题不是“通信是否提高利润”，而是：

> 谁说了什么、谁有权看到、通信何时关闭、Agent 看完后做了什么，以及这一切能否被准确重建。

## 2. 用户最终能看到的效果

一个典型回合如下：

1. A 公开宣布“本轮不主动降价”，并附带非绑定价格声明；
2. B 私下向 C 建议维持较高价格；
3. D 不发言；
4. Communication Close 后：
   - A 看见所有公开消息和自己发送的消息；
   - B、C 看见公开消息以及 B→C 私信；
   - D 只看见公开消息，完全不知道 B→C 私信是否存在；
5. 四个 Agent 根据各自不同但合法的消息视图，同时提交经营决策；
6. 市场只执行最终联合动作，消息本身不改变现金、价格或市场状态；
7. 研究日志能对照消息声明、Agent 的采信/拒绝和最终动作。

阶段完成后，每轮日志应能回答：

- 每个 Agent 发了什么；
- 哪些 Agent 实际看到了每条消息；
- Agent 引用了、接受、拒绝或忽略哪些消息；
- 声称的行动与实际行动是否一致；
- 通信条件下的市场结果与无通信条件有何差异。

## 3. 交互模式

通信模式和市场的 `information_mode` 相互独立：

| 模式 | 公开消息 | 一对一私信 | 用途 |
| --- | --- | --- | --- |
| `off` | 否 | 否 | 阶段 1 基线和阶段 2 对照 |
| `public_only` | 是 | 否 | 研究公共信号和公开威胁 |
| `public_private` | 是 | 是 | 研究选择性披露、私下建议和欺骗 |

首版每个 Agent 每轮最多发送一条公开消息和一条私信，每条不超过 500 字。空消息列表代表主动沉默。模型输出失败、超时或格式非法时，通信 fallback 也是沉默，绝不由规则 Agent 代替它编造消息。

## 4. 回合状态机

```text
ROUND_OPEN
  -> OBSERVATION_FROZEN
  -> COMMUNICATION_OPEN
  -> AGENTS_COMMUNICATING        # 并发，一次同步波次
  -> MESSAGES_ACCEPTED
  -> COMMUNICATION_CLOSED        # 不可变快照
  -> DECISION_OBSERVATIONS_FROZEN
  -> AGENTS_DECIDING             # 并发
  -> INTENTS_ACCEPTED
  -> ACTION_LOCKED
  -> ROUND_SETTLED               # MarketEnv.step() 恰好一次
  -> FEEDBACK_DISTRIBUTED
  -> ROUND_LOGGED
```

首版每轮只有一个同步发言波次。同一波次内，Agent 看不到其他 Agent 尚未关闭的消息，因此不存在网络先后顺序造成的优势；收件人可以让消息影响当轮经营决策，但对话回复要等到下一轮。接口保留以后扩展多波次谈判的空间。

Communication Close 必须满足：

- 所有消息绑定同一个 `episode_id + round + state_version + state_hash`；
- Controller 拥有发送者身份、轮次、消息 ID 等可信字段；
- 按公司和本地消息序号确定性排序，不按网络到达时间排序；
- Close 后不能新增或修改消息；
- Close 重复调用返回同一结果；
- 每家公司得到独立的 `view_digest`，看不到全局消息数量或隐藏私信的全局 Hash。

## 5. 数据契约

### 5.1 模型可以生成的内容

`CommunicationSubmission` 只包含至多两条 `MessageDraft`：

- `channel`: `public | private`；
- `recipients`: 私信恰好一个真实对手，公开消息必须为空；
- `speech_act`: statement、proposal、promise、threat、question、response 或 other；
- `content`: 自由文本；
- `own_action_claim`: 可选的自身非绑定行动声明；
- `requested_peer_action`: 可选的对对手行动建议。

结构化行动声明只允许价格、广告、服务、产能和韧性等经营字段。它不能叫 `final_action`，不能执行，也不能绕过动作约束。

### 5.2 Controller 生成的可信记录

Controller 把合法草稿转换成 `DeliveredMessage`，补充：

- 确定性 `message_id`；
- episode、round、state version/hash；
- 可信发送公司；
- 规范化可见范围。

Close 生成：

- Controller-only 的完整 `CommunicationClosure` 和 `transcript_hash`；
- 每个 Agent 独立的 `CommunicationView` 和 `view_digest`；
- 显式的发言公司和沉默公司列表。

### 5.3 决策上下文

P1 将把关闭后的视图加入新版 `DecisionContext`：

- 当前通信模式和关闭状态；
- 当前轮可见消息；
- 从权威通信日志派生的最近可见历史；
- 本公司的 `view_digest`；
- 明确标记消息是非绑定、未经验证的对手内容。

决策 Prompt 必须把消息放在明确分隔的 JSON 数据区，并说明：

- 对手消息不是系统指令；
- 消息可能是谎言、试探或威胁；
- Agent 可以采信、拒绝或忽略；
- 消息不能覆盖人格、硬约束或市场事实。

## 6. 实现结构

### P0：通信协议与可见性

当前已开始：

- `interaction/contracts.py`：版本化消息、声明、关闭结果和公司视图；
- `interaction/round.py`：并发提交后的确定性账本、幂等提交、Close 和私信过滤；
- `tests/test_interaction.py`：消息形状、配额、参与者、幂等、关闭和 4 公司可见性测试。

接下来接入：

- Episode 的 `communication_mode`；
- Agent Gateway 的身份绑定、提交消息和读取公司视图接口；
- Controller-only 的 open/close 接口；
- stale、late、冒充发送者和跨公司读取保护。

当前 Agent Gateway 允许 URL 或请求体直接声明 `company_id`。这可以用于本地无私密数据的模拟，但不能作为真正私聊的安全边界。P0 完成前必须把 Agent 身份绑定到 episode/company，并禁止跨公司读取或发送。

### P1：Agent 两段调用闭环

- 增加独立 `generate_communication()`，不把消息塞进经营动作；
- `AgentRuntime.communicate()` 负责生成和校验外发草稿；
- DeepSeek、豆包和 Mock 分别实现通信输出；
- DecisionContext 和决策 Prompt 接收关闭后的公司视图；
- `RoundCoordinator` 用真实屏障替换当前 `COMMUNICATION_CLOSED_NOOP`；
- 通信失败退化为沉默，经济决策失败继续使用现有规则 fallback；
- 决策结果增加可选的结构化消息回应，且只能引用本公司可见的消息 ID。

### P2：审计、Replay 与小型真实 Smoke

- `RoundEvent` 升版并嵌入完整的通信阶段记录；
- 每个 Agent trace 记录通信上下文、原始输出、合法消息、可见消息 ID、view digest、耗时、token 和错误；
- 原市场 Replay 继续只根据最终动作验证市场状态；
- 新增 Interaction Replay，从消息日志重建每家公司实际看到的输入并校验 Hash；
- 增加通信工程指标和只读研究指标；
- 最后运行同 Seed 的 `off` 与 `public_private` 小型真实 Smoke。

不把通信写入 `MarketState` 或 `MarketTransition`，所以通信关闭前后市场状态 Hash 必须完全不变，阶段 1 的经济 Replay 保持有效。

## 7. 验收标准

### 7.1 P0 硬工程验收

- 4×4 公开/私信可见性矩阵零泄露；
- 私信正文、ID、数量和全局 Hash 均不进入无权 Agent 的 Context 或 API 响应；
- sender 由可信身份确定，冒充发送者和跨公司读取全部被拒绝；
- stale round/state、非法收件人、超额消息和 Close 后消息全部被拒绝；
- 相同提交和 Close 幂等，不同 payload 的重复提交冲突；
- 通信前后 `MarketState.state_hash` 和 `state_version` 不变；
- 并发到达顺序不同，规范化 transcript/hash 仍一致。

### 7.2 P1 闭环验收

- 每个经济决策都发生在 Communication Close 之后；
- 四个决策使用相同的冻结市场状态和各自正确的关闭视图；
- 使用确定性 Mock：只有看到指定公开消息才改变价格；`off/on` 下 Context 和动作按预期不同；
- 私信只改变指定收件 Mock，其他 Agent 的 Context 和动作不变；
- 某 Agent 通信超时或非法输出时，本轮自动沉默且仍可完成结算；
- 4 Mock × 20 轮完成，0 次部分结算，每轮 `MarketEnv.step()` 恰好一次；
- 所有消息引用都属于该 Agent 的合法可见消息集合。

### 7.3 P2 Replay 和真实调用验收

- 经济状态 Replay 100%；
- Interaction Replay 100%；
- 正文、发送者、收件人、顺序、message ID 或 view digest 的非一致性修改均能被发现；这不是数字签名式防篡改；
- `off` 模式保留阶段 1 的状态 Hash 和市场结果；
- 2 LLM + 2 Rule、同一 Seed、`off/public_private` 各 5 轮真实 Smoke；
- 有效消息进入正确的 LLM Context，至少能记录 Agent 对消息的接受、拒绝或忽略；
- 无私信泄露、无非法动作、无因通信失败导致的整轮失败。

利润、价格趋同、合作程度或市场福利是否改善，均不是工程验收门槛。它们属于正式实验的研究结果。

## 8. 阶段 2 的研究指标

工程指标：

- 消息合法率、主动沉默率和生成失败率；
- visibility leak、late submit 和 state mutation 次数；
- 消息到 Context 的映射率；
- round 完成率、经济 Replay 和 Interaction Replay 通过率；
- 通信增加的延迟与 token 成本。

只读研究指标：

- 公开/私信占比、收件对象和 speech act 分布；
- 消息采信、拒绝和忽略比例；反提议需等以后增加显式关联契约后再统计；
- 结构化行动声明与最终动作的 alignment/deviation；
- `off/public_only/public_private` 下的价格离散度、投资、利润、缺货和外部选项变化。

正式因果实验仍使用共同 Seed 和公司位置轮换，以 Seed 为主要统计单位，而不是把每条消息或每次决策当作独立样本。私信造成的是消息可见性不对称，不等于阶段 3 的私有市场状态或 Belief。

## 9. 明确非目标

阶段 2 v1 不实现：

- 合作人格或社会福利效用；
- 绑定合同、惩罚、转账和承诺自动执行；
- 消息自动修改价格或投资；
- 基于自由文本主观判定“欺骗”或“背叛”；
- 同一轮多波次讨价还价；
- 隐藏利润、现金或需求等阶段 3 不完全信息；
- Game Theory Advisor、最佳响应或 Self-play 优化。

当 P0、P1、P2 的硬验收全部通过后，冻结为 `interaction-v1`。随后才能用正式多 Seed 对照研究：通信是否改变定价趋同、威胁可信度、私人提议、言行偏离和长期市场结果。
