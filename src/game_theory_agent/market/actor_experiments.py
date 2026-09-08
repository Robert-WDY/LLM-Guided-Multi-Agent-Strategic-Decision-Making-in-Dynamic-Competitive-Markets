"""Resumable four-actor experiments with durable per-call journals."""
import json
import os
import math
from dataclasses import asdict
from pathlib import Path
from . import MarketEnv, MarketState, MarketConfig
from .actor_policies import actor_ids, options, observation, objective, descriptions, actions_for, reward
from .actor_model import choose_option,ActorModelValidationError
from .replay import EpisodeManifest, MarketTransition, verify_replay
from .protocols import sha256_hash
from game_theory_agent.game_theory.market_strategies import MODES as THEORY_MODES
from game_theory_agent.game_theory.objectives import ADVISOR_MODES
from game_theory_agent.game_theory.advisor_robust import ROBUST_MODES


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authorize_schema_retry(directory,round_number,actor,reason):
    """Explicit operator recovery, never called automatically by the worker."""
    path=Path(directory)/f"round-{round_number:03d}.json";journal=read_json(path);old=journal["decisions"][actor]
    if old["status"]!="failed_schema" or old.get("reviewed_attempts"):raise ValueError("only one reviewed retry of a known completed schema failure is allowed")
    old["reviewed_attempts"]=[dict(reason=reason,failed_response=old.get("model_response"),raw_recorded=bool(old.get("model_response")))]
    old["status"]="retry_authorized";write_json(path,journal)


def learning_choice(actor,memory):
    m=memory.get(actor,{});counts=m.get("counts",{});means=m.get("means",{})
    for option in options(actor):
        if not counts.get(option):return option
    return max(options(actor),key=lambda o:means[o]+math.sqrt(2*math.log(sum(counts.values())+1)/counts[o]))


def update_memory(memory,state,choices):
    for actor,option in choices.items():
        m=memory.setdefault(actor,dict(counts={},means={}))
        n=m["counts"].get(option,0)+1;v=reward(state,actor);v=v/(abs(v)+10_000_000)
        m["means"][option]=m["means"].get(option,0)+(v-m["means"].get(option,0))/n;m["counts"][option]=n


async def run_episode(config, *, directory, seed, rounds=10, modes=None, fixed=None, memory=None,
                      authorize_real=False, on_progress=None, cancelled=None, model_name="deepseek-v4-flash",company_count=4):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);cp=directory/"checkpoint.json"
    modes=dict(modes or {});fixed=dict(fixed or {})
    identity=dict(config_hash=config.config_sha256,seed=seed,rounds=rounds,modes=modes,fixed=fixed,model=model_name)
    if type(company_count)!=int or not 2<=company_count<=10:raise ValueError('company count must be 2..10')
    if company_count!=4:identity['company_count']=company_count
    if cp.exists():
        saved=read_json(cp)
        if saved["identity"]!=identity:raise ValueError("experiment checkpoint belongs to a different configuration")
        env=MarketEnv(config);s=MarketState.from_dict(saved["state"]);env.load_state(s)
        manifest=EpisodeManifest.create(env,MarketState.from_dict(saved["manifest"]["initial_state"]),cooperation_mode="combined_v1")
        if manifest.to_dict()!=saved["manifest"]:raise ValueError("experiment manifest mismatch")
        traces=[MarketTransition.from_dict(t) for t in saved["transitions"]];memory=saved["memory"]
    else:
        env=MarketEnv(config);s=env.reset(company_ids=[f'company_{chr(65+i)}' for i in range(company_count)],episode_id=f"four-actor-common-{seed}",episode_seed=seed,max_rounds=rounds,cooperation_mode="combined_v1")
        manifest=EpisodeManifest.create(env,s,cooperation_mode="combined_v1");traces=[];memory=json.loads(json.dumps(memory or {}))
        write_json(cp,dict(identity=identity,config=config.to_dict(),state=s.to_dict(),manifest=manifest.to_dict(),transitions=[],memory=memory))
    ids=actor_ids(s)
    if set(modes)-set(ids) or set(fixed)-set(ids):raise ValueError("unknown experiment actor")
    if any(m not in ("rule","learning","model","neural",*THEORY_MODES,*ADVISOR_MODES,*ROBUST_MODES) for m in modes.values()):raise ValueError("unknown actor control")
    if any(m in (*THEORY_MODES,*ADVISOR_MODES,*ROBUST_MODES) and a not in s.company_ids for a,m in modes.items()):raise ValueError("game-theory policies support companies only")
    for actor,choice in fixed.items():
        if choice not in options(actor):raise ValueError("unknown fixed portfolio")
    if "model" in modes.values() and not authorize_real:raise ValueError("real model authorization required")
    while not s.terminal:
        if cancelled and cancelled():return dict(status="cancelled",state=s.to_dict(),memory=memory)
        journal_path=directory/f"round-{s.round:03d}.json"
        journal=read_json(journal_path) if journal_path.exists() else dict(state_hash=s.state_hash,decisions={})
        if journal["state_hash"]!=s.state_hash:raise ValueError("decision journal state mismatch")
        choices={}
        for actor in ids:
            if cancelled and cancelled():return dict(status="cancelled",state=s.to_dict(),memory=memory)
            existing=journal["decisions"].get(actor)
            if existing and existing["status"]!="retry_authorized":
                if existing["status"]!="complete":raise RuntimeError("previous paid call outcome unknown; no automatic retry")
                if 'policy_memory_after' in existing:memory.setdefault('_theory',{})[actor]=existing['policy_memory_after']
                choices[actor]=existing["option"];continue
            mode=modes.get(actor,"rule")
            record=dict(mode=mode,observation=observation(s,actor),objective=objective(actor),options=descriptions(actor))
            if existing:record["reviewed_attempts"]=existing["reviewed_attempts"]
            if actor in fixed:choice=fixed[actor];record["mode"]="fixed"
            elif mode in ROBUST_MODES:
                from game_theory_agent.game_theory.advisor_beliefs import RobustRequest
                from game_theory_agent.game_theory.advisor_robust import advise
                choice='balanced'
                if actor in s.strategic_market.active_company_ids:
                    detail=advise(config,s,RobustRequest(company_id=actor,goal=ROBUST_MODES[mode],history=memory.get('_advisor_public_history',[]),horizon=1,scenarios=2,max_candidates=8,finalists=1,step_budget=max(500,len(ids)*50),diagnostics=False,backtest=False,seed=seed%2147483648))
                    record.update(advisor_decision=detail,advisor_recipe=detail['recipe'],objective=detail['objective'],execution='v17 response-aware recipe overrides bookkeeping option')
                else:record['execution']='exited company: frozen action'
            elif mode in ADVISOR_MODES:
                from game_theory_agent.game_theory.advisor_search import advise
                from game_theory_agent.game_theory.objectives import AdviceRequest
                choice='balanced'
                if actor in s.strategic_market.active_company_ids:
                    detail=advise(config,s,AdviceRequest(company_id=actor,goal=ADVISOR_MODES[mode],horizon=2,scenarios=2,max_candidates=8,step_budget=200,diagnostics=False,backtest=False,seed=seed%2147483648))
                    record.update(advisor_decision=detail,advisor_recipe=detail['recipe'],objective=detail['objective'],execution='recipe overrides balanced bookkeeping option')
                else:record['execution']='exited company: frozen baseline action'
            elif mode in THEORY_MODES:
                from game_theory_agent.game_theory.market_strategies import choose
                choice,detail,policy_memory=choose(config,s,actor,mode,memory.get('_theory',{}).get(actor))
                memory.setdefault('_theory',{})[actor]=policy_memory
                record.update(theory_decision=detail,policy_memory_after=policy_memory)
            elif mode=="learning":choice=learning_choice(actor,memory)
            elif mode=="neural":
                from .neural_policy import predict
                choice=predict(actor,record["observation"],config=config)
            elif mode=="model":
                journal["decisions"][actor]={**record,"status":"pending_unknown"};write_json(journal_path,journal)
                try:
                    d=await choose_option(actor_id=actor,objective=record["objective"],observation=record["observation"],
                                          options=record["options"],max_output_tokens=128,model_name=model_name)
                except ActorModelValidationError as exc:
                    journal["decisions"][actor]={**record,"status":"failed_schema","model_response":exc.record}
                    write_json(journal_path,journal);raise
                record["model_response"]=asdict(d);choice=d.option_id
            else:
                choice=learning_choice(actor,memory) if actor=="government" else "balanced"
            journal["decisions"][actor]={**record,"status":"complete","option":choice};write_json(journal_path,journal)
            choices[actor]=choice
        aa=actions_for(config,s,choices)
        for actor,decision in journal['decisions'].items():
            if 'advisor_recipe' in decision:
                from game_theory_agent.game_theory.advisor_market import recipe_action
                aa[actor]=recipe_action(config,s,actor,decision['advisor_recipe'])
        extra={a:o for a,o in choices.items() if a not in s.company_ids}
        result=env.step(f"{s.episode_id}:{s.round}:{s.state_version}",aa,actor_choices=extra)
        if any(m in ROBUST_MODES for m in modes.values()):
            from game_theory_agent.game_theory.advisor_beliefs import public_frame
            memory.setdefault('_advisor_public_history',[]).append(public_frame(s))
        traces.append(MarketTransition.create(s,aa,result));s=result.state_after;update_memory(memory,s,choices)
        journal.update(settled_state_hash=s.state_hash,rewards={a:reward(s,a) for a in ids})
        write_json(journal_path,journal)
        write_json(cp,dict(identity=identity,config=config.to_dict(),state=s.to_dict(),manifest=manifest.to_dict(),
                          transitions=[t.to_dict() for t in traces],memory=memory))
        if on_progress:on_progress(s.state_version,rounds)
    verify_replay(MarketEnv(config),manifest,traces)
    result=dict(status="complete",seed=seed,rounds=rounds,welfare_cents=s.welfare_accounting.cumulative_total_economic_welfare_cents,
                company_profit_cents=sum(c.financial.cumulative_profit_cents for c in s.companies),
                supplier_profit_cents=sum(v.cumulative_profit_cents for v in s.supply_chain.suppliers),
                consumer_surplus_cents=s.welfare_accounting.cumulative_consumer_surplus_cents,
                government_cash_cents=s.government.cash_cents,memory=memory,state_hash=s.state_hash,replay_passed=True)
    if company_count!=4:result['company_count']=company_count
    write_json(directory/"summary.json",result);return result
