"""Frozen market, experiments and read-only response-learning data boundaries."""
import gzip,hashlib,json,math,time
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
from game_theory_agent.market import MarketConfig,MarketEnv,MarketState,CompanyAction,load_market_config
from game_theory_agent.market.actor_experiments import read_json,write_json
from game_theory_agent.market.actor_policies import company_action
from game_theory_agent.game_theory.market_strategies import choose
from game_theory_agent.game_theory.reliable_advisor import ReliableRequest,advise,rebind,local_candidates
from game_theory_agent.game_theory.reliable_response import frame,features,bucket,fit,predict,MODEL_PATH
from game_theory_agent.game_theory.advisor_audit import audit_settlement,paired_summary
from game_theory_agent.game_theory.objectives import objective,score
from game_theory_agent.local_budget import status

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/reliable-advisor-v19';BASE=load_market_config(ROOT/'configs/market_v14_local.yaml')
REGIMES=('normal','recession','tight_supply','price_sensitive','project_feasible','scaled')
SPEC=dict(regimes=list(REGIMES),companies=[2,5,10],goals=['profit','welfare'],seeds=list(range(521001,521021)),
          training_seeds=list(range(519001,519025)),calibration_seeds=list(range(520001,520009)),cooperation_seeds=list(range(522001,522021)),scaling_seeds=list(range(523001,523021)),
          candidates=12,discovery_scenarios=3,validation_scenarios=8,horizons=[3,5],budget=1800,
          modes=['none','generic','conditional'],main_baseline='agent_draft',secondary_baseline='rule',
          gate='Per-cell mean paired 95% bootstrap lower>0 and realized negative rate<=0.1; abstention is not improvement.',
          paid_gate='All four experiment suites complete, engineering passes, conditional model held-out Brier no worse than generic, and conditional advice passes every 36 primary 3-round cell and pooled 5-round negative<=10%. Otherwise paid phase remains blocked.',
          seed_independence='20 seeds per cell; pooled intervals average conditions within seed; no multiple-comparison correction',
          real_models=['deepseek-v4-flash','doubao-seed-2-0-lite-260215'],real_seeds=list(range(526001,526007)),real_decision_units=36,
          market_changes='Configuration and external institution only; core market hashes frozen')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    temp.replace(path)
def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def freeze_market():
    paths=[*ROOT.glob('src/game_theory_agent/market/*.py'),*ROOT.glob('configs/market*.yaml'),ROOT/'src/game_theory_agent/decisioning.py',ROOT/'src/game_theory_agent/gameplay.py',ROOT/'src/game_theory_agent/agents/observation.py',ROOT/'src/game_theory_agent/strategic_reliability/public_rollout.py']
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths};p=OUT/'frozen-market.json'
    if p.exists():assert read_json(p)==manifest,'frozen market changed'
    else:write_json(p,manifest)
    p=OUT/'preregistration.json'
    if p.exists():assert read_json(p)==SPEC
    else:write_json(p,SPEC)
    return manifest
def make(m,n,seed,rounds=20):
    raw=BASE.to_dict()
    if m=='recession':raw['market']['base_demand_orders']=7200
    elif m=='tight_supply':
        for s in raw['supply_chain']['suppliers'].values():s['base_capacity_orders']=3500
    elif m=='price_sensitive':
        for s in raw['consumer_choice']['segments'].values():s['coefficients_ppm']['price']*=2
    elif m=='project_feasible':raw['strategic_market']['threshold_project']['required_total_contribution_cents']=n*300000
    elif m=='scaled':
        raw['market']['base_demand_orders']=3000*n
        for s in raw['supply_chain']['suppliers'].values():s['base_capacity_orders']=2750*n
        raw['supply_chain']['transaction_accounting']['initial_supplier_cash_cents']=5000000*n
        raw['autonomous_market']['government']['initial_cash_cents']=12500000*n
        # Demand orders scale consumer population; cohort shares/budgets per consumer are unchanged.
        raw['strategic_market']['threshold_project']['required_total_contribution_cents']=2000000*n
    elif m!='normal':raise ValueError(m)
    c=MarketConfig.from_mapping(raw);e=MarketEnv(c);s=e.reset(company_ids=[f'company_{chr(65+i)}' for i in range(n)],episode_id=f'reliable-{seed}',episode_seed=seed,max_rounds=rounds,cooperation_mode='combined_v1');return c,e,s
def actions(config,s,memory,seed):
    aa={};updated={}
    for i,cid in enumerate(s.company_ids):
        mode=('theory_tft','theory_grim','theory_wsls')[(seed+i)%3]
        option,_,mem=choose(config,s,cid,mode,memory.get(cid));updated[cid]=mem
        aa[cid]=company_action(config,s,cid,option)
    return aa,updated
def step(e,s,aa):return e.step(f'{s.episode_id}:{s.round}:{s.state_version}',aa).state_after
def source(m,n,seed):
    c,e,s=make(m,n,seed);memory={};history=[]
    for _ in range(4):
        aa,memory=actions(c,s,memory,seed);history.append(dict(state=frame(s),actions={k:v.to_dict() for k,v in aa.items()}));s=step(e,s,aa)
    aa,_=actions(c,s,memory,seed)
    return c,s,memory,aa['company_A'],history
def rollout(c,state,memory,seed,first,draft,horizon=5):
    e=MarketEnv(c);e.load_state(state);s=state;memory=deepcopy(memory);start=s.company('company_A');profit=welfare=0.;minimum=start.financial.cash_balance_cents;rows=[];metrics={};checks=0
    for t in range(min(horizon,s.rounds_remaining)):
        aa,memory=actions(c,s,memory,seed);aa['company_A']=rebind(c,s,'company_A',first if t==0 else draft);a=step(e,s,aa);checks+=audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());x=a.company('company_A');profit+=.95**t*x.financial.round_profit_cents;welfare+=.95**t*a.welfare_accounting.round_total_economic_welfare_cents;minimum=min(minimum,x.financial.cash_balance_cents)
        rows.append(dict(round=s.round,actions={k:v.to_dict() for k,v in aa.items()},state_hash=a.state_hash));s=a
        metrics[t+1]=dict(profit=profit,welfare=welfare,cash=x.financial.cash_balance_cents-start.financial.cash_balance_cents,share=x.commercial.market_share_ppm-start.commercial.market_share_ppm,capacity=x.operations.base_capacity_orders-start.operations.base_capacity_orders,resilience=x.risk.resilience_ppm-start.risk.resilience_ppm,minimum_cash=minimum,exited='company_A' not in s.strategic_market.active_company_ids)
    return dict(metrics=metrics,transitions=rows,checks=checks)

def replay(c,state,records):
    e=MarketEnv(c);e.load_state(state);s=state;checks=0
    for row in records:
        after=step(e,s,{a:CompanyAction.from_dict(v) for a,v in row['actions'].items()});assert after.state_hash==row['state_hash'];checks+=audit_settlement(s.to_dict(),after.to_dict(),c.to_dict());s=after
    return checks
