"""Build reviewable Markdown from verified primary records, retaining failures."""
from statistics import stdev
from report_reliable_v20 import *
from evaluate_reliable_v201 import DEST201

def money(x):return '—' if x is None else f'{x/100:,.2f}'
def pct(x):return '—' if x is None else f'{100*x:.2f}%'

def table(lines,heading,goals):
    lines.extend([f'## {heading}','', '|目标 / 方法|案例|实际干预|覆盖率|干预负例|负例率|全案例平均增益（合成元）|95%种子簇区间|最差增益（合成元）|','|---|---:|---:|---:|---:|---:|---:|---|---:|'])
    for g,v in goals.items():
        for method,key in [('原v19','v19'),('本阶段修正','v20')]:
            x=v[key];ci=x['cluster_bootstrap_mean_ci95_cents']
            lines.append(f'|{g} / {method}|{x["cases"]}|{x["interventions"]}|{pct(x["coverage"])}|{x["losses"]}|{pct(x["intervention_loss_rate"])}|{money(x["all_case_mean_gain_cents"])}|[{money(ci[0])}, {money(ci[1])}]|{money(x["worst_gain_cents"])}|')
    lines.extend(['','同一个种子下的不同人数/市场条件相关，置信区间按种子簇重采样。全案例均值包含弃权的零增益；干预由经济动作实际变化核对。数值均为模拟市场金额，不是人民币经营预测。',''])

def main():
    a=read_json(DEST/'analysis.json');b=read_json(DEST201/'analysis.json');diagnosis=a['diagnosis']['results']
    lines=['# v20 / v20.1 可靠性修复与独立验收（2026-09-09）','',
      '**工程实现与实验链路已完成；经济可靠性未全面通过，两个新版本均保留为实验性选项。** v20增加状态压力与误差预算，但相同覆盖率表现未改善，未推广。v20.1删除不成功的压力重排、保留原候选排序并进行独立误差校准；利润负例率改善，但严格风险上界仍略高于门槛，福利仍不合格。',
      '', '## 实现与验收范围','',
      '- 核对v19全量720案例的真实干预、利润/福利风险、尾部损失；全部28负例及21个严格匹配正例进入2×2诊断。',
      '- 每例8条独立未来路径；A公开状态/预测策略，B真实状态/预测策略，C公开状态/实际记忆策略，D真实状态/实际记忆策略。保持候选不变，不重搜；D重现原动作和结算hash。真实状态只在离线诊断使用。',
      '- 49例×7项初态组件敏感性，每组4条共同未来路径。对手现金、容量、品牌、风险、历史、市场参数、供应商现金分别检查；没有把不一致的历史账本拼接为有效状态。',
      '- v20：180新校准状态（30种子簇）、180新留出状态（30簇）、60局A/B/C六轮滚动实验（10簇），另在180留出状态比较none/generic两种简单响应模型。',
      '- v20.1：180再次独立校准状态（30簇）、300新确认状态（50簇）、60局新A/B/C六轮滚动实验（10簇）。两阶段滚动合计360条分支、2160轮，不是360个独立种子。',
      '- 新界面显示误差预算、调整后余量、弃权原因；支持保存恢复，超出校准范围明确弃权。原v19、市场公式与冻结ZIP保持不变。',
      '', '## 已修复的评价口径','',
      '利润269/360实际干预、9负例（3.35%）；福利90/360实际干预、19负例（21.11%）。disposition与经济动作比较零冲突。旧报告的28/720会把弃权混入风险分母；现在同时报告全案例收益、干预负例率、损失幅度、覆盖率与同覆盖率比较。','',
      '## 状态×策略诊断结论','', '|组别|49例平均绝对预测误差（合成元）|','|---|---:|']
    for k in 'ABCD':lines.append(f'|{k}|{money(mean(abs(r["errors_cents"][k]) for r in diagnosis))}|')
    lines.extend(['','修正初态比只替换策略明显降低该诊断集的误差。但这些是经过失败选择的开发案例，不能外推为整体市场误差比例。单因素品牌修正有改善，无法消除主要偏差；多项未知初态及其交互仍未被准确识别。D为零是执行一致性控制，不是现实正确性证明。',
                  '', '早期随机路径批次误把原生规则风格seed也改变，已修正为冻结策略身份、只改未来市场随机输入；旧批次完整保留为pre-policy-seed-fix，不计主结果。',
                  '', '[逐案例误差和福利预测分项](reliable-v20-error-isolation.md)。'])
    table(lines,'v20：首次修复留出结果',a['holdout']['goals'])
    table(lines,'v20.1：新种子确认结果',b['holdout']['goals'])
    lines.extend(['## 相同覆盖率比较','', '|阶段|目标|干预数|旧排序负例|新筛选负例|旧全案例均值|新全案例均值|','|---|---|---:|---:|---:|---:|---:|'])
    for stage,result in [('v20',a),('v20.1',b)]:
        for g,v in result['holdout']['goals'].items():
            for x in v['equal_coverage']:lines.append(f'|{stage}|{g}|{x["interventions"]}|{x["v19"]["losses"]}|{x["v20"]["losses"]}|{money(x["v19"]["all_case_mean_gain_cents"])}|{money(x["v20"]["all_case_mean_gain_cents"])}|')
    lines.extend(['','按预测分数选同样数量的建议，未使用实际收益排序。v20.1保留原候选排序；其跨期限误差预算仍可能改变跨案例接受次序。风险筛选的收益/覆盖率权衡不等于搜索能力提高。',
                  '', '## 六轮持续互动','', '|阶段|目标|条件|相对每轮无建议均值（合成元）|95%种子簇区间|有干预的局数|负收益局数|','|---|---|---|---:|---|---:|---:|'])
    for stage,result in [('v20',a),('v20.1',b)]:
        for g,v in result['rolling'].items():
            for mode in ('once','every'):
                x=v[mode];ci=x['cluster_bootstrap_mean_ci95_cents'];lines.append(f'|{stage}|{g}|{mode}|{money(x["all_case_mean_gain_cents"])}|[{money(ci[0])}, {money(ci[1])}]|{x["interventions"]}|{x["losses"]}|')
    lines.extend(['','A每轮从新状态生成原计划，B只在第一轮修正，C每轮生成新计划再修正/弃权。三条分支的对手记忆独立更新；同轮看不到未执行候选。实验只让company_A采用建议，不代表全体公司共同采用或真实LLM长期互动。10个滚动种子簇只支持初步方向性判断。',
                  '', '## 为什么没有宣布全面修复','', '|目标|v20.1均值下界>0|覆盖率门槛|干预种子簇|有损失簇|簇级任一损失概率95%上界|风险门槛≤10%|','|---|---|---|---:|---:|---:|---|'])
    for g,v in b['holdout']['goals'].items():
        x=v['v20'];lines.append(f'|{g}|{v["gate"]["positive_mean"]}|{v["gate"]["coverage"]}|{x["intervention_seed_clusters"]}|{x["seed_clusters_with_any_loss"]}|{pct(x["seed_any_loss_upper95"])}|{v["gate"]["risk_upper"]}|')
    lines.extend(['','该簇级上界比单次建议负例率更保守，不能把两者当成同一个概率。零失败的bootstrap区间会退化，所以不能用“0到0”宣称没有风险。利润的单次负例率较低，但一次亏损仍可较大；福利建议稀少且仍有亏损。持续互动还出现新的利润负局，不能用单次干预验收替代长期验证。',
                  '', '## 简单模型对照与合作制度',''])
    if 'response_ablations' in a:
        lines.extend(['|目标|响应模型|实际干预|负例|全案例平均增益（合成元）|','|---|---|---:|---:|---:|'])
        for g,group in a['response_ablations'].items():
            for mode,x in group.items():lines.append(f'|{g}|{mode}|{x["interventions"]}|{x["losses"]}|{money(x["all_case_mean_gain_cents"])}|')
    lines.extend(['','合作结果复用已修正的1080局：完整按可达性、人数、制度、背离条件统计在analysis.json的cooperation.groups。诚实暂停不再算违约，旧46800条承诺检查保留；逐对象信誉提高识别不等于消除搭便车利润优势。没有新增无必要的大矩阵。',
                  '', '## 工程验收、费用与使用','',
                  '- 独立动作回放：v20 420份/4680步/189606算术检查；v20.1 540份/5880步/238553算术检查。全部通过，并核对每轮原计划确实重新生成、同轮对手不预知建议。',
                  '- Python回归和新增专项覆盖530项：完整513项回归中6项日期依赖失败，已隔离测试日期并定向复测通过，另有新增门禁/API专项。生产价格过期阻断未放宽。前端19项、类型检查、lint和生产构建通过。',
                  '- 浏览器实际执行v20.1、自动保存5032921d记录、刷新重新读取通过；模型验证为正但因第1轮未校准而明确保留原计划。最后一轮服务重启被自动审批拒绝（仅返回blocked by policy），现有服务仍运行；最终构建中的说明文字与新增报告入口需下次重启载入。',
                  '- 新增真实模型调用0次；账本仍186条历史调用、累计上限30元、保守预留9.993565元。原全目标收益门槛未通过，价格确认也已过期，未绕过收费门禁。',
                  '- 本机入口：http://localhost:3210/ → 博弈策略实验室 → v20.1经验风险筛选（实验性）。仅支持已校准配置、company_A、2/5/10公司、纯利润/福利、固定风险参数和观测范围。其他情况保留原计划。',
                  '', '## 证据与复现','',
                  '- `runs/reliable-advisor-v20/`：原风险表、49例误差诊断、7项敏感性、首次校准/留出/滚动、简单响应消融、原错误批次及校验。',
                  '- `runs/reliable-advisor-v201/`：新校准、300例确认、60局滚动、独立回放与分析。',
                  '- `configs/reliable-calibration-v20.json` 与 `configs/reliable-calibration-v201.json`：固定经验误差预算及支持范围。',
                  '- 冻结后验收：`implementation-before-holdout.json` / `frozen.json`。研究者真实状态不进入服务模块。',
                  '', '尚未解决：公开信息下联合隐藏状态辨识、福利预测的低覆盖率/尾部损失、持续干预产生的分布变化，以及真实模型采纳环节。不能通过继续调同一留出集或关闭全部建议把这些问题记为解决。'])
    (ROOT/'docs/reliable-v20-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    detail=['# v20状态×策略误差隔离','', '单位均为合成元；误差=预测均值−D权威均值。每例固定原计划和选中候选，8条未来随机路径。','', '|案例|A误差|B误差|C误差|D误差|A−D配对误差95%路径区间|','|---|---:|---:|---:|---:|---|']
    for r in diagnosis:
        data=read(DEST/'diagnosis'/(r['case_id']+'.json.gz'));diff=[x['gain']-y['gain'] for x,y in zip(data['groups']['A'],data['groups']['D'])];spread=2.365*stdev(diff)/math.sqrt(8)
        detail.append('|'+r['case_id']+'|'+ '|'.join(money(r['errors_cents'][k]) for k in 'ABCD')+f'|[{money(mean(diff)-spread)}, {money(mean(diff)+spread)}]|')
    detail.extend(['','这些路径区间仅描述该固定案例的模拟期望差，不是总体因果效应或现实风险区间。','', '## 组件敏感性','', '|修正组件|平均绝对误差（合成元）|','|---|---:|'])
    for k,v in a['sensitivity']['mean_absolute_error_cents'].items():detail.append(f'|{k}|{money(v)}|')
    detail.extend(['','## 逐例预测/权威福利分项','', '正数表示候选相对原计划增加该项目。外部成本项增加应从总福利扣除；转移收入不是净福利增量。'])
    components=['round_consumer_surplus_cents','round_downstream_producer_surplus_cents','round_upstream_producer_surplus_cents','round_government_net_budget_cents','round_stockout_externality_cents','round_business_exit_externality_cents','round_total_economic_welfare_cents']
    for r in diagnosis:
        data=read(DEST/'diagnosis'/(r['case_id']+'.json.gz'));detail.extend(['',f'### {r["case_id"]}','','|福利分项|A预测增量|D权威增量|误差|','|---|---:|---:|---:|'])
        for k in components:
            x=mean(v['components'][k] for v in data['groups']['A']);y=mean(v['components'][k] for v in data['groups']['D']);detail.append(f'|{k}|{money(x)}|{money(y)}|{money(x-y)}|')
    (ROOT/'docs/reliable-v20-error-isolation.md').write_text('\n'.join(detail)+'\n',encoding='utf-8')

if __name__=='__main__':main()
