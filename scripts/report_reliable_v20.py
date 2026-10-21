"""Read-only risk, equal-coverage, diagnosis and closed-loop acceptance report."""
from evaluate_reliable_v20 import *
from collections import Counter

def upper_binomial(k,n,alpha=.05):
    if not n or k==n:return 1.
    lo,hi=0.,1.
    for _ in range(70):
        p=(lo+hi)/2;cdf=sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))
        if cdf>alpha:lo=p
        else:hi=p
    return (lo+hi)/2

def risk(rows):
    r=stats(rows);clusters=defaultdict(list)
    for x in rows:
        if x['intervention']:clusters[x['seed']].append(x['gain'])
    k=sum(any(v<0 for v in values) for values in clusters.values());n=len(clusters)
    r.update(intervention_seed_clusters=n,seed_clusters_with_any_loss=k,seed_any_loss_upper95=upper_binomial(k,n))
    # Paired condition dependence is preserved when bootstrapping loss rates.
    grouped=defaultdict(list)
    for x in rows:grouped[x['seed']].append(x)
    rng=random.Random(2009);blocks=list(grouped.values());rates=[]
    for _ in range(1500):
        chosen=[x for _ in blocks for x in rng.choice(blocks) if x['intervention']]
        if chosen:rates.append(sum(x['gain']<0 for x in chosen)/len(chosen))
    rates.sort();r['cluster_bootstrap_intervention_loss_ci95']=[rates[int(.025*len(rates))],rates[min(len(rates)-1,int(.975*len(rates)))]] if rates else None
    r['loss_interval_note']='A zero-loss bootstrap interval is degenerate; use the conservative seed-any-loss upper bound as the release gate.'
    return r

def outcomes(split):
    cal=load_calibration();rows=[];counter=Counter()
    for p in sorted((DEST/split).glob('*.gz')):
        v=read(p);s=v['summary'];r=apply_calibration(v['proposal'],s['config_sha256'],cal);e=r['empirical_calibration'];changed=economic_change(r['action'],v['draft'])
        score_old=min((x['lower_bound'] for x in v['proposal']['validation']),default=-1e30)
        score_new=e['adjusted_margin'] if e['adjusted_margin'] is not None else -1e30
        rows.append(dict(**s,new_intervention=changed,new_gain=s['gain'] if changed else 0,old_margin=score_old,new_margin=score_new,new_eligible=e['calibrated']))
        counter.update(e['reasons'])
    result={}
    for goal in ('profit','welfare'):
        rr=[r for r in rows if r['goal']==goal]
        old=risk(rr);new=risk([{**r,'intervention':r['new_intervention'],'gain':r['new_gain']} for r in rr]);curve=[]
        # Equal-coverage selection uses only predicted scores. All outcomes remain in denominator.
        old_order=sorted([r for r in rr if r['intervention']],key=lambda r:(-r['old_margin'],r['case_id']))
        new_order=sorted([r for r in rr if r['intervention'] and r['new_eligible']],key=lambda r:(-r['new_margin'],r['case_id']))
        counts=sorted({new['interventions'],*[int(len(rr)*q) for q in (.05,.1,.2,.3)]})
        for k in counts:
            if k<1 or k>min(len(old_order),len(new_order)):continue
            item=dict(interventions=k,coverage=k/len(rr))
            for name,order in [('v19',old_order),('v20',new_order)]:
                selected={r['case_id'] for r in order[:k]};item[name]=risk([{**r,'intervention':r['case_id'] in selected,'gain':r['gain'] if r['case_id'] in selected else 0} for r in rr])
            curve.append(item)
        result[goal]=dict(v19=old,v20=new,equal_coverage=curve,gate=dict(positive_mean=new['cluster_bootstrap_mean_ci95_cents'][0]>0,
                coverage=new['coverage']>=(.1 if goal=='profit' else .05),risk_upper=new['seed_any_loss_upper95']<=.1))
        result[goal]['passed']=all(result[goal]['gate'].values())
    return dict(goals=result,reasons=dict(counter),rows=rows)

def cooperation():
    rows=read_json(OUT/'cooperation.json')['results'];groups=[]
    for n in (2,5,10):
        for inst in sorted({r['institution'] for r in rows}):
            for threshold in ('reachable','unreachable'):
                for defection in sorted({r['defection'] for r in rows}):
                    rr=[r for r in rows if r['companies']==n and r['institution']==inst and r['threshold']==threshold and r['defection']==defection]
                    if not rr:continue
                    groups.append(dict(companies=n,institution=inst,threshold=threshold,defection=defection,cases=len(rr),project_success_rate=mean(r['project_success'] for r in rr),
                          good_acceptance=mean(r['good_acceptance'] for r in rr if r['good_acceptance'] is not None) if any(r['good_acceptance'] is not None for r in rr) else None,bad_acceptance=mean(r['bad_acceptance'] for r in rr if r['bad_acceptance'] is not None) if any(r['bad_acceptance'] is not None for r in rr) else None))
    return dict(source='runs/reliable-advisor-v19/cooperation.json',source_sha256=sha(OUT/'cooperation.json'),cases=len(rows),groups=groups,scope='Reuse only the corrected main batch; no extra institution experiment or causal claim.')

def main():
    result=dict(v19=read_json(DEST/'intervention-risk-v19.json')['results'],diagnosis=read_json(DEST/'diagnosis.json'),sensitivity=read_json(DEST/'state-sensitivity-extended.json'),cooperation=cooperation())
    for split in ('calibration','holdout'):
        if (DEST/(split+'.json')).exists():result[split]=outcomes(split)
    if (DEST/'rolling.json').exists():
        rows=read_json(DEST/'rolling.json')['results'];rolling={}
        for g in ('profit','welfare'):
            rr=[r for r in rows if r['goal']==g];entry={}
            for mode in ('once','every'):
                entry[mode]=risk([{**r,'intervention':r[mode+'_interventions']>0,'gain':r[mode+'_gain']} for r in rr])
            entry['every_minus_once']=stats([{**r,'intervention':True,'gain':r['every_gain']-r['once_gain']} for r in rr]);entry['positive_closed_loop_mean']=entry['every']['cluster_bootstrap_mean_ci95_cents'][0]>0
            rolling[g]=entry
        result['rolling']=rolling
    result['paid_phase']=dict(new_calls=0,status='blocked_original_preregistered_gate',cumulative_budget_cny=30,price_confirmation_expired=True)
    if (DEST/'response-ablations.json').exists():
        ablations=read_json(DEST/'response-ablations.json')['results'];table={}
        for g in ('profit','welfare'):
            table[g]={}
            for mode in ('none','generic'):
                rr=[dict(seed=r['seed'],**next(x for x in r['results'] if x['mode']==mode)) for r in ablations if r['goal']==g]
                table[g][mode]=risk(rr)
        result['response_ablations']=table
    write_json(DEST/'analysis.json',result)
    if 'holdout' in result:
        print(json.dumps({g:result['holdout']['goals'][g] for g in ('profit','welfare')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
