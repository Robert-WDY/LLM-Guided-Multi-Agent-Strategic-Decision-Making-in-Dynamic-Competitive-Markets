"""Exercise the running local API; preserve idempotent records and free queue jobs."""
import json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from game_theory_agent.local_budget import status
from game_theory_agent.market.actor_experiments import write_json
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'runs/advisor-v17'

def call(path,payload=None):
 request=Request('http://127.0.0.1:8010/api/v1/controller/'+path,data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
 with urlopen(request,timeout=300) as reply:return json.load(reply)

def main():
 before=status();records=[]
 for goal in ('profit','welfare'):
  request=dict(request_id='v17-live-'+goal,kind='advisor',parameters=dict(advisor_version='v17',company_count=5,goal=goal,seed=376001,horizon=1,scenarios=2,max_candidates=12,step_budget=1200,diagnostics=False))
  result=call('theory-lab/experiments',request);assert result==call('theory-lab/experiments',request)==call('theory-lab/experiments/'+result['id'])
  assert result['result']['v17']['response_complete'];assert result['result']['version']=='advisor-v17.0.0';records.append(result['id'])
 episodes=call('saved-episodes')['episodes'];public_history=None
 if episodes:
  episode=episodes[0]['episode_id'];source=call('episodes/'+episode+'/export');request=dict(request_id='v17-live-history',kind='advisor',parameters=dict(advisor_version='v17',episode_id=episode,horizon=1,scenarios=2,max_candidates=8,step_budget=1200,diagnostics=False,backtest=False))
  result=call('theory-lab/experiments',request);r=result['result'];expected=[t for t in source['transitions'] if t['state_before']['round']<r['source_round']][-60:]
  assert len(r['request']['history'])==len(expected);records.append(result['id']);public_history=len(expected)
 bad=dict(request_id='v17-live-invalid',kind='advisor',parameters=dict(advisor_version='v17',goal='custom',weights={'profit':0.5}))
 try:call('theory-lab/experiments',bad);raise AssertionError('bad weights accepted')
 except HTTPError as exc:assert exc.code==422
 req=dict(request_id='v17-live-all-companies',seeds=[376101],rounds=5,company_count=2,variants=[dict(label='rule'),dict(label='v17-profit',modes={c:'robust_profit' for c in ('company_A','company_B')}),dict(label='v17-welfare',modes={c:'robust_welfare' for c in ('company_A','company_B')})])
 job=call('workbench/jobs',req);assert call('workbench/jobs',req)['id']==job['id'];print(json.dumps(dict(job=job['id'],records=records)),flush=True)
 for _ in range(300):
  current=next(j for j in call('workbench/jobs')['jobs'] if j['id']==job['id'])
  if current['status']=='complete':break
  assert current['status'] in ('queued','running'),current['error'];time.sleep(2)
 else:raise AssertionError('queue did not complete in ten minutes')
 export=call('workbench/jobs/'+job['id']+'/export');assert len(export['cases'])==3
 checked=0
 for case in export['cases'].values():
  for journal in case['decisions']:
   for cid,d in journal['decisions'].items():
    if 'advisor_decision' not in d:continue
    r=d['advisor_decision'];assert r['version']=='advisor-v17.0.0' and r['v17']['response_complete'];checked+=1
 assert checked==20 and status()==before
 with urlopen('http://localhost:3210/',timeout=30) as reply:assert reply.status==200
 report=dict(passed=True,records=records,public_history_samples=public_history,invalid_weights_rejected=True,idempotence=True,job_id=job['id'],episodes=3,rounds=15,advisor_decisions=checked,api_persistence=True,frontend_http=True,new_real_calls=0,browser_qa=False)
 write_json(DEST/'live-api.json',report);print(json.dumps(report))

if __name__=='__main__':main()
