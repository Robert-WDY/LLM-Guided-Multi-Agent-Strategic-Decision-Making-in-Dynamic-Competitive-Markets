"""Recalculate interventions over the entire contract, including delayed legal execution."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta
from study import World,RAW,OUT,read,save


def audit_file(file):
    data=read(file)
    rows=data if isinstance(data,list) else [data]
    output=[]
    for r in rows:
        arm=r['arm']
        if arm not in ('conditional','generic','wide','diagnostic','gated','learned','gate0','gate1','short','feedback'):continue
        plans=[d for d in r['contracts'] if d['start']>6 and d.get('active',True)]
        immediate=sum(d.get('actual_override',False) for d in plans)
        effective=immediate
        doubtful={d['start']:d for d in plans if r['market']=='native' and abs(d['price']-d['draft'])>1.e-8
                  and (not d.get('actual_override',False) or (r['condition']=='demand' and d['start']==r.get('when')))}
        if doubtful:
            w=World('native',r['seed'],r['condition'])
            for contract in r['contracts']:
                if w.t>max(doubtful):break
                if w.t in doubtful:
                    clone=w.clone()
                    alt=[clone.execute(contract['draft'])[0] for _ in range(contract['end']-contract['start']+1)]
                    selected=r['rows'][contract['start']-1:contract['end']]
                    effective+=int(any(v!=u[0] for v,u in zip(alt,selected)))-int(contract['actual_override'])
                for t in range(contract['start'],contract['end']+1):
                    row=w.execute(contract['price'])
                    assert row==r['rows'][t-1]
        stage='realunits' if file.name.startswith('realunits-') else 'real2' if file.name.startswith('real2-') else 'long' if file.name.startswith('long-') else 'primary'
        provider=file.name.split('-')[1] if stage.startswith('real') else ''
        value=sum(.99**i*x[3] for i,x in enumerate(r['rows']))
        output.append((stage,r['market'],provider,r['seed'],r['condition'],arm,len(plans),immediate,effective,value))
    return output


if __name__=='__main__':
    files=[f for pattern in ('confirm-*.gz','long-confirm-*.gz','real2-*.gz','realunits-*.gz') for f in RAW.glob(pattern)]
    with ProcessPoolExecutor(max_workers=9) as ex:
        rows=[v for group in ex.map(audit_file,sorted(files),chunksize=8) for v in group]
    baselines={(stage,m,p,s,c):v for stage,m,p,s,c,a,_,_,_,v in rows if a=='feedback'}
    groups=defaultdict(list)
    for stage,m,p,s,c,a,n,i,e,v in rows:
        if a=='feedback':continue
        base=baselines[('real2' if stage=='realunits' else stage,m,p,s,c)]
        groups['/'.join((stage,m,p,a))].append((s,n,i,e,v-base))
    out={}
    for key,values in groups.items():
        exposed={s for s,n,i,e,d in values if e}
        harmed={s for s,n,i,e,d in values if e and d < -1.e-8}
        n,k=len(exposed),len(harmed);den=sum(x[1] for x in values)
        out[key]=dict(opportunities=den,immediate=sum(x[2] for x in values),full=sum(x[3] for x in values),
                      coverage=sum(x[3] for x in values)/den,exposed=n,harmed=k,
                      upper95=None if n==0 else 1 if n==k else beta.ppf(.95,k+1,n-k))
    save(OUT/'term-exposure.json',out)

