"""Readable acceptance report generated from all evaluation outcomes."""
from statistics import mean
from analyze_market_v18 import interval
from evaluate_market_v18 import *

NAMES=dict(normal='原市场',recession='需求收缩',boom='需求增长',tight_supply='供给紧张',cash_stress='资金紧张',price_sensitive='价格敏感',quality_sensitive='质量敏感',disruption='频繁冲击',project_feasible='可达公共项目',scaled='资源随人数扩展',asymmetric='产能不对称')

def main():
 r=read_json(OUT/'analysis.json');stable=read_json(OUT/'stability.json');accept=read_json(OUT/'acceptance.json');groups=r['groups'];advice=r['advisor_groups'];v17=[a for a in advice if a['variant']=='v17'];fast=[a for a in advice if a['variant']=='fast'];get=lambda m,n,p:next(x for x in groups if (x['market'],x['companies'],x['policy'])==(m,n,p))
 lines=['# 合作、竞争、市场丰富度与建议质量：v18全面评测','',
 '本次评测完成，但“市场完美模拟现实、建议在所有情况下稳定有效”不是验收结论。工程正确性、决策空间、经济收益、计算延迟、现实校准分别判定，负面结果全部保留。生产市场参数未被调整成对实验有利的值。','',
 f"完成 {r['tournament_episodes']} 局×20轮={r['tournament_rounds']:,}轮群体实验、{r['advisor_cases']}组源状态/目标评测（{r['advisor_decisions']}次建议）、54局540轮共同采用建议、990条五轮动作探针及132条合同期限补充轨迹、18组重复/扰动测试及27次单轮串行计时与3次实际界面默认参数计时。最后用修补后的独立审计重新回放所有主群体和闭环记录，{r['arithmetic_checks']:,}次算术核对通过；{accept['backend_tests']}项后端回归通过。",'',
 f"v17按预注册收益门槛通过 {sum(x['effectiveness_pass'] for x in v17)}/{len(v17)} 个市场×人数×目标分组；相对Agent原策略通过 {sum(x['incumbent_pass'] for x in v17)}/{len(v17)} 组。快速预设相对规则通过 {sum(x['effectiveness_pass'] for x in fast)}/{len(fast)} 组。通过条件为按8个独立种子重采样的均值增益95%区间下限>0，且负收益比例≤10%。每组8例，1次损失即12.5%，因此无法通过10%门槛；这是当前样本粒度的限制。未通过不一定证明无效，但不能宣称可靠优于基线。",'',
 '默认合作贡献与项目门槛不匹配：原市场全合作24例项目全失败，可达门槛对照24例全成功。固定总资源下10家企业的规则平均利润率为负，按人数扩展资源后2/5/10家约7%—8%；规模和资源稀释必须分开分析。', '',
 '速度也有配置边界：单轮标准/快速配置全部达到预注册的3/5/10秒样本门槛，但界面默认三轮预测加诊断的2/5/10公司实测约8.81/49.07/132.29秒，尚不满足快速交互的目标。完整默认配置不能用快速实验配置的成绩代替。', '',
 '## 设计与可信度','',
 '- 11个市场、2/5/10家公司、8个独立种子、9种群体策略。每格20轮；同种子同初始状态，不把回合、公司数或压力条件当成独立样本。','- 群体包括规则、合作贡献、搭便车、一名背离者、互惠、价格竞争、双方价格协调、双方产能互助、四方UCB学习。合作/价格协调/互助动作真实进入结算。','- 这批群体是脚本策略和在线有限选项学习，不能称为新LLM自由谈判、自发形成联盟。消费者是群体决策，供应商固定两家，企业数才从2变化到10。','- 11个市场覆盖需求±、供给/现金压力、价格/质量偏好、冲击频率、公共项目门槛、每公司资源规模归一化、两两1000/6000产能不对称。原市场保留，门槛可达版本明确作为反事实对照。','- 建议评测用6类市场、8个新种子、两种目标。合作、价格战和UCB对手先运行4轮，再独立提交同轮真实动作；建议器只获合法信息，评价器在其实际动作下执行建议。','- 对照包括继续Agent原策略、规则动作、v16、v17标准与快速预设；另用最多48个实际合法动作加所有被测建议/原动作构造事后参照。研究者真状态参照不传给建议器；参照只限当前一轮有限动作，不代表动态全局最优。','- 标准v16/v17统一1轮、2情景、12候选、1200步。快速v17为8候选、1个非基线入围者、随人数500—700步。此轮未修改建议目标或按收益调参。','- 所有区间为探索性百分位bootstrap，未校正多重比较；每个市场人数目标只有8个独立种子。汇总时先在种子内平均，不能把864次建议说成864个独立样本。','',
 '## 合作与竞争是否真正进入市场','',
 '下表为各格8种子的描述统计。合作−搭便车福利差的单位是合成元。互助订单为实际交付；罚款为实际扣款。项目成功率比较原市场与专门的可达门槛条件，不能据此宣称原配置本来就能支持成功合作。','',
 '| 市场/公司数 | 合作−搭便车福利 [95%区间] | 全合作项目成功 | 互助实际订单/局 | 价格协调罚款/局（元） | 协调策略退出企业总数 |','|---|---|---:|---:|---:|---:|']
 for m in MARKETS:
  for n in (2,5,10):
   c=next(x for x in r['cooperation_contrasts'] if (x['market'],x['companies'],x['treatment'])==(m,n,'cooperators'));co=get(m,n,'cooperators');aid=get(m,n,'mutual_aid');cartel=get(m,n,'coordination')
   lines.append(f"| {NAMES[m]}/{n} | {interval(c['welfare'])} | {co['project_successes']}/8 | {aid['mean_aid_orders']:.0f} | {cartel['mean_fines']/100:,.0f} | {cartel['exits']} |")
 lines+=['','利润、消费者剩余和福利可以反向变化。合作不能由贡献额定义为成功；价格协调不能被当作社会合作的同义词。单个搭便车者的优势，以及互惠是否抑制其优势，均在analysis.json的cooperation_contrasts中按同种子列出。当前互惠规则只要求至少一名其他企业上一轮有贡献：因此多人中一名背离者可能不会触发集体惩罚。这是该制度的可检验限制。','',
 '原市场两家公司时，企业A从合作改为搭便车，20轮平均多获30,251元；另一家公司采取当前互惠规则后，A的利润再减少9,413元。到了原市场5家公司，互惠与无条件合作两种对手下，A利润和总福利在8个种子完全相同，说明当前规则没有识别和惩罚这一名背离者。可达项目两家公司时，A搭便车反而平均少获79,166元；可达项目10家公司时则平均多获52,801元、总福利下降约289,188元。这些结果显示关键贡献者、搭便车与集体利益冲突会随人数和门槛而变，不能把所有市场条件统称为同一种囚徒困境。数值均为配对样本均值，具体区间和反例保留。','',
 '## 市场是否给足决策空间','',
 '66组固定场景，每组15类动作、连续5轮，只改变企业A策略，其余策略规则相同并对状态演化作反应。按利润、福利、销量、产能、韧性、公共项目等经济结果判断有效，不用动作ID或状态hash充当经济效果。','', '| 动作 | 66组中改变经济结果的组数 |','|---|---:|']
 labels=['price_down','price_up','service','advertising','capacity','resilience','public','project','procure_less','procure_more','diversify','resilient_supplier','contract_long','reserve']
 for label in labels:lines.append(f"| {label} | {sum(label in x['economically_effective'] for x in r['probes']['results'])}/66 |")
 lines+=['','补充核对发现contract_long在66例都与默认3轮相同，原探针不能用于判定合同无效。另做1轮对3轮的66组五轮配对：66组动作不同，其中56组经济结果不同（132条补充轨迹），原始零结果保留。这里的“有效”仅表示能改变经济结果，不表示获利或优于基线。采购量上限、现金上限、终局规则可能使两个意图修正成同一动作；公共项目门槛和固定贡献策略也可能令项目不可达。5轮探针尚不能检验所有长期创新、进入退出和动态契约。','',
 '## 建议在真实对手动作下是否有帮助','',
 '| 市场/人数/目标 | v17相对规则：收益下降例/8 | v17收益门槛 | 相对原Agent策略门槛 | v17平均有限参照遗憾（元） | 快速预设收益门槛 |','|---|---:|---|---|---:|---|']
 for x in v17:
  f=next(f for f in fast if (f['market'],f['companies'],f['goal'])==(x['market'],x['companies'],x['goal']))
  lines.append(f"| {NAMES[x['market']]}/{x['companies']}/{x['goal']} | {x['vs_rule']['negative']}/{x['n']} | {'通过' if x['effectiveness_pass'] else '未通过'} | {'通过' if x['incumbent_pass'] else '未通过'} | {x['mean_oracle_regret']/100:,.0f} | {'通过' if f['effectiveness_pass'] else '未通过'} |")
 lines+=['','总体描述及按种子聚类的收益区间：','', '| 版本/目标 | 相对规则收益下降例/条件数 | 相对原策略收益下降例 | 按种子平均增益 [95%区间]（元） |','|---|---:|---:|---|']
 for k,x in r['pooled'].items():lines.append(f"| {k} | {x['losses']}/{x['cases']} | {x['incumbent_losses']} | {interval(x['seed_clustered_gain'])} |")
 lines+=['','在288个源状态/目标案例中，v16、v17和快速版分别有28、143、141例达到本次有限参照中的最高收益。有限参照遗憾=本次评测允许动作中最好的实际收益−该建议收益；使用的是固定对手本轮动作，不能解释成存在可预知的稳赚替代策略，更不是证明多人纳什均衡。原状态暖启动退出造成的不可用案例单独保留：'+str(len(r['unavailable']))+'例。','',
 '一个可追查的同种子例子：482001、两家公司、合作型对手，v17在原市场选择提价10%并保留现金，比规则多获68,544.32元；只改变为需求收缩市场，同样建议反而比规则少获2,245.47元，比原Agent策略少245.47元。两次内部验证都认为改善且反击检查完成，实际负例说明对手/需求预测集合仍不充分。金额均为合成市场；不能把验证全部正样本当成真实收益保证。详见case-studies.json。', '',
 '## 全部企业同时采用建议','',
 '正常、供给紧张、可达项目三类市场×3种人数×2个新种子×规则/仅A建议/全企业建议，共54局10轮。其他角色继续自身规则。每轮同时读取结算前状态；全企业建议在这里采用利润目标，不能期待自动维持社会合作。','',
 '| 市场/人数 | 仅A建议：A利润变化均值（元） | 全企业建议：企业总利润变化（元） | 全企业建议：福利变化（元） | 全企业退出数 |','|---|---:|---:|---:|---:|']
 for m in SPEC['closed_loop_markets']:
  for n in (2,5,10):
   f=[x for x in r['closed_loop'] if (x['market'],x['companies'],x['policy'])==(m,n,'focal_v17')];allrows=[x for x in r['closed_loop'] if (x['market'],x['companies'],x['policy'])==(m,n,'all_v17')]
   lines.append(f"| {NAMES[m]}/{n} | {mean(x['focal_profit_delta'] for x in f)/100:,.0f} | {mean(x['profit_delta'] for x in allrows)/100:,.0f} | {mean(x['welfare_delta'] for x in allrows)/100:,.0f} | {sum(x['exits'] for x in allrows)} |")
 lines+=['','这里只用两个独立种子，属于长局工程和方向性结果，不能宣称一般均衡或收敛。','',
 '## 稳定性和速度','',
 f"18组同输入重复，除耗时外全部结果逐项一致。72个自身价格/现金±1%扰动中，{stable['recipe_changes']}次改变建议配方。配方变化不自动等于不稳定：它也可能是合法的目标/约束边界响应；本轮记录变化，不强制掩盖变化。",'',
 '| 公司数 | 版本 | 中位秒数 | 样本p95秒数 | 预注册上限 | 速度验收 |','|---:|---|---:|---:|---:|---|']
 for x in r['latency']:lines.append(f"| {x['companies']} | {x['variant']} | {x['median']:.3f} | {x['p95']:.3f} | {x['limit']} | {'通过' if x['passed'] else '未通过'} |")
 lines+=['','计时在批量任务结束后串行运行，含完整建议调用，排除HTTP/UI传输。每组仅3次，样本p95近似最大值，不是生产延迟承诺。快速预设应结合上面的收益门槛判断，不能仅因速度更快就自动替换标准建议。','', '| 公司数 | 界面默认H3/S3/16候选+诊断+事后检验：秒 | 反击检查完整 |','|---|---:|---|']
 for x in read_json(OUT/'ui-default-latency.json')['results']:lines.append(f"| {x['companies']} | {x['seconds']:.3f} | {x['complete']} |")
 lines+=['','| 公司数 | 长局实际建议次数 | 中位秒 | 实测p95秒 | 最大秒 |','|---|---:|---:|---:|---:|']
 for n,x in r['closed_loop_workload_latency'].items():lines.append(f"| {n} | {x['calls']} | {x['median']:.3f} | {x['p95']:.3f} | {x['maximum']:.3f} |")
 lines+=['','长局计时包含四进程并行时的资源争用与演化后的市场状态，属于实际实验负载观察，不能与串行初始状态时间混为一谈。界面默认参数每个规模只测1次；这是实测观察，不是P95。不能用上面的单轮快速时间代表完整默认计算。','',
 '## 与真实世界对齐到什么程度','',
 '采用独立来源进行结构与量级对照：竞争应同时看价格、质量和消费者结果，而不仅看企业利润。[OECD竞争评估](https://www.oecd.org/en/topics/sub-issues/competitive-and-fair-markets/competition-assessment.html)。供给瓶颈会传播，预防性囤货也可能加剧短缺，本轮通过供给压力与采购/库存策略检验相应方向。[BIS Bulletin 61](https://www.bis.org/publ/bisbull61.pdf)。合作依赖制度和利益冲突处理，不能预设人人合作一定更好。[Ostrom研究介绍](https://www.nobelprize.org/prizes/economic-sciences/2009/press-release/?wptouch_preview_theme=enabled)。','',
 '外部金额参照来自美国Census CB26-96（2026-06-08发布），为大型零售企业季调税后利润/销售额；不是本机市场的训练数据。计算利润率如下：[Census原始发布](https://www.census.gov/econ/qfr/retail/current/index.html)。','', '| 期间 | 外部税后利润率 |','|---|---:|']
 for k,x in r['external_retail_margins'].items():lines.append(f'| {k} | {x:.2%} |')
 lines+=['','| 原市场规则公司数 | 模拟企业经营利润/收入均值 |','|---|---:|']
 for n in (2,5,10):lines.append(f"| {n} | {get('normal',n,'rule')['mean_margin']:.2%} |")
 lines+=['','税后利润与模型经营利润口径不同，行业、企业规模、时间单位也不一致，不能对两列直接作拟合优度或宣称校准通过。当前仅有结构合理性与外部数量级警示；缺少行业实测需求弹性、工资成本、成本分布、融资数据、外部性估计和独立现实留出样本，现实校准验收仍未通过。','',
 '## 真实语言模型证据','',
 '本轮新增付费调用0次，原10元累计预算保持不变。当前批量模型选项并不等同完整自然语言协调器。重新读取并校验了3份历史摘要的SHA256：旧版两模型6局真实合作链路曾全部履约；另有每格3例的历史承诺/背叛/声誉与提示敏感性实验。历史模型之间和提示条件之间存在差异，摘要记录只能作为已有方向性证据，不能充作本轮新模型实验或现实验证。详见historical-real-evidence.json。','',
 '## 修复与失败记录','',
 '1. 互助审计漏算提供方服务费：补上互助订单×费率、商品收入+服务费核对，并注入1分错误验证。','2. 退出企业的学习/组合动作重新进入在营价格护栏：normal/5公司/481001/price_war第18轮，连balanced请求也会抬高冻结价而被拒绝。退出后直接返回冻结动作；四种组合回归一致。初期归因只提到value/margin改价不够准确，此处根据旧函数复现更正，详见exit-guard-reproduction及regression-reproductions。','3. 清算现金被误当经营利润：用显式配置独立复算折旧与清算舍入，核对资产转现金；不把现金回收加为营业利润。','',
 '上述修补均由保留的失败案例触发，不修改主实验种子或经济目标。最终所有主轨迹用补齐后的审计再次重放；留出前后实现hash及修补记录分别保存，不能声称整个过程从未修代码。','',
 '## 验收范围与后续重点','',
 '通过的部分是市场能实际结算多种合作竞争行为、状态边界修补、经济结果可审计和实验可复算。建议效果和速度按各组实测结果判定。市场丰富不代表所有策略都有收益，尤其应检查公共项目可达性、互惠对少数背离者是否失效、价格协调对消费者的代价，以及人数增长造成的资源稀释。','',
 '还需要推进的方向：优先使合作项目门槛、行动预算和收益分配相匹配；建议筛选加入Agent正在准备的动作作为对照，并加强未见对手和长期投入后果验证；优化默认诊断/多轮预测的计算与交互，不能仅把预算调小就宣称质量不变；按具体行业识别和验证需求/成本；让LLM在统一完整动作协议下自由提议、接受、违约并进行新的配对实验；扩展消费者与供应商主体数；检验长期投资/企业进入/动态融资；针对未通过的建议场景研究目标或对手模型，再用全新种子复验。现有结果不足以支持“完美模拟真实市场”或“任何规模都给出好建议”。','',
 '原始证据位于runs/market-evaluation-v18：preregistration、压缩逐轮轨迹、所有候选/建议、独立回放、稳定性、串行计时、来源与完整统计。通过evaluation-v18证据包冻结，旧v17及更早发布包保留。','']
 path=ROOT/'docs/market-evaluation-v18-results.md';path.write_text('\n'.join(lines),encoding='utf-8');print(path)

if __name__=='__main__':main()
