# Game Theory Lab：多智能体博弈实验控制台

日期：2026-08-21。

## 定位

前端从“单公司经营游戏页”重构为 `Multi-Agent Game Research Dashboard`。它同时面向实验操作者、研究观察者和 Human Agent，并遵循三条边界：

1. 权威研究市场公式仍只存在于 Python `MarketEnv`；前端的确定性演示函数仅用于交互检查，并明确标为非研究证据；
2. 不显示 Chain of Thought，只显示结构化 Decision Summary、输入 Hash、候选 Advice、Final Action 与可重建结果；
3. Research Demo 必须有显式标识，不能冒充真实 Episode、真实 LLM 调用或研究证据。

## 八个工作区

工作区不再直接充当网站首页。新的主入口先区分三种使用意图：个人体验、观察实验和研究控制台。三者都经过环境配置，但进入后拥有不同导航：个人体验只保留实时现场与回合记录；观察模式提供现场、Agent、通信、市场与 Replay；研究模式才显示完整 Game Theory 工具。任意子界面均可返回主页，已有会话会在主页显示“继续当前会话”。

- 实验配置：信息模式、市场、轮数、共同 Seed、Communication、Shared Resilience、Game Theory Stack 和 2–4 个 Agent；
- 实时博弈：从第 1 回合开始的单页决策舱，在一个页面汇总 Public/Private Observation、可见消息、Belief、Opponent Model、Plan、Advisor、Human Action、四家公司竞争和最近 Joint Settlement；
- Agent 观察站：Public/Private/Hidden Observation、Belief、Planning 和 Decision Summary；
- 通信中心：公开/私信过滤、可见性矩阵和 Proposal→Acceptance→Commitment→Actual→Fulfillment；
- 信念与策略：Action Belief、Opponent Model、Utility Inference、Advisor 候选与 Agent 是否采纳；
- 市场面板：利润轨迹、份额与广告/服务/产能/韧性投入；
- Replay 调试器：True State→Observation→Belief→Communication→Advisor→Final Action→Market Result；
- 实验报告：仅在当前 Episode 完成后展示本次最终结果与证据边界。

当前 Episode 的“实验报告”入口实行完成门禁：Round 1 至最终回合期间完全隐藏，只有 Episode Complete 后才出现。报告内容使用当前 Agent 的最终份额、利润、韧性与份额守恒结果；Demo 报告明确说明它不是 LLM 研究证据。历史多 Seed 研究结论应由单独的实验档案入口读取，不能提前放进正在进行的 Episode。

## Persona 与 Human Agent

Agent 卡支持 Human、Doubao、DeepSeek 和 Rule 驱动，并提供 Balanced、Aggressive、Risk Guarded、Long-term Selfish、Cooperator、Free Rider 与 Retaliator Persona Profile。Drawer 展示效用权重、时间折扣、风险厌恶、合作和机会主义参数；这些值是实验配置，不被描述为模型真实心理。

Human Agent 页面可以设置价格、广告和共享韧性贡献，也可以输入策略摘要。当前“自然语言转换 Action”按钮只提供界面入口，尚未绑定新的转换 API；数值 Action 与后端安全护栏仍是权威执行路径。

## 真实后端与演示边界

页面启动时只探测 Market API 健康状态。普通 Episode 可直接使用现有 `/api/episodes`；开启 Communication、Cooperation、Belief、Opponent Model、Utility 或 Advisor 时，用户必须在本地输入 Controller Token。Token 只保存在 React 内存，不写入 URL、Local Storage 或演示数据。

创建真实高级 Episode 后，前端不会绕过 Communication Close 和 Agent Intent 直接调用市场 Step，而是提示等待 Coordinator 完成屏障和 Settlement。当前前端完成了高级 Episode 创建与权威 MarketState hydration，下一后端阶段才需要增加安全的 Experiment/Timeline/Trace 聚合读 API 和 Coordinator 控制 API。

后端离线或未输入 Token 时可加载 Research Demo。演示从 Round 1 开始，最大回合结算后进入 `EPISODE COMPLETE`，不会循环。每次推进读取 Human 的价格、广告与贡献，结合对手回合策略、历史份额和确定性冲击计算演示订单吸引力，再把四家公司份额归一化为 100%。这修复了固定加减造成的单调份额假象，并让低价扩大份额但可能损害利润等基本权衡可以在 UI 中观察。该函数不是 `MarketEnv`，不能进入实验报告或替代后端 RoundEvent。

实时页不再要求参与者在多个导航页面间拼接信息。信息按真实处理顺序放在同一页。公司摘要只保留市场份额、变化、公开价格和抗冲击能力，删除卡片底部难以阅读的完整动作串。对手现金、成本、利润和人格仍保持遮蔽；“同页全部信息”指智能体合法可见并与决策相关的全部输入，而不是突破不完全信息边界。

随后根据用户反馈进一步改为全中文的六步过程：接收信息、交流、形成判断、制定策略、做出决策、市场结算。研究员或参与者可以选择 A/B/C/D，逐步查看该公司收到的公共与私有上下文、可见交流、公开证据形成的对手判断、策略目标、最终动作和结算影响。页面不展示自由形式隐藏推理，只展示结构化输入、依据和结果。本轮消息与对手判断改为始终可见的小型摘要，不再需要点击展开；个人动作和结算占据更大的主区域。

第一回合的对手判断遵循冷启动规则：没有已经结算的公开历史时，显示“尚无历史证据，暂不判断”，不显示 33% 等伪精确先验。第一轮结算后，演示才根据各公司的公开价格变化和份额变化生成两条证据，并更新下一轮价格动作概率。正式后端仍以版本化对手判断状态和可见信息重建为准。

## 视觉与工程验收

- `npm run lint` 通过；
- `npm run build` 通过；
- 服务端 HTML 测试通过；
- 桌面浏览器检查 Experiment Setup、Persona Drawer、Live、Belief、Replay；
- Persona Drawer 的 Preset 与全部权重可见；
- Live 对手现金和利润显示为隐藏；
- Live 初始回合为 1，Round 20 完成后按钮锁定且不会返回 Round 1；
- 四家公司份额每轮合计 100.0%，连续五轮中存在方向反转而不是机械单调轨迹；
- Human 低价与高价反事实产生不同份额和利润；
- Replay 节点切换会更新 Trace Inspector；
- 页面导航后自动回到顶部，避免从 Setup 底部进入 Live 时跳过回合屏障；
- 首屏只显示三个角色入口，不显示实验内部导航或未完成报告；
- 三个入口均先到环境配置页，并可从配置页、运行页返回主页；
- 观察模式无 Human Action，研究工具不会混入个人体验导航；
- 实验报告在 Round 20 完成前隐藏，完成后显示当前 Episode 总结；
- 768×900 窄屏无横向溢出；
- 浏览器控制台 error/warning 为 0。

## 下一步接口

若要让前端完全驱动真实多 LLM 实验，后端下一步应提供受保护、面向研究控制台的聚合接口，而不是让浏览器拼装内部 Ledger：Experiment CRUD、Coordinator Round Control、Agent-scoped Trace、Round Timeline、Episode Report 和落盘 Replay 查询。私信和 Private State 仍必须由公司身份或研究员审计权限过滤，不能用请求体自报 Company ID。
