import asyncio,json
from dataclasses import asdict,replace
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
from game_theory_agent.market import MarketEnv,MarketState,load_market_config
from game_theory_agent.market.replay import EpisodeManifest,MarketTransition,verify_replay
from game_theory_agent.market.actor_model import choose_option
from game_theory_agent.market.government_strategy import OPTIONS,unpack
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage23-government"
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def actions(config,s):
    return {cid:replace(build_rule_action(config,s,cid),resilience_budget_cents=100000 if s.rounds_remaining>1 else 0) for cid in s.company_ids}
def step(env,s,aa,option):
    return env.step(f"{s.episode_id}:{s.round}:{s.state_version}",aa,actor_choices={"government":option} if option else {})
async def main():
    OUT.mkdir(parents=True,exist_ok=False);load_dotenv(ROOT/".env",override=False)
    config=load_market_config(ROOT/"configs/market_v14_stage23.yaml")
    write("preregistration.json",dict(config=config.to_dict(),config_hash=config.config_sha256,seeds=list(range(230201,230209)),
        policies=[*OPTIONS,"learning"],rounds=20,paired="same episode_id and seed for every treatment",
        company_policy="rule plus 100000 cents verified resilience each non-terminal round",
        real_seeds=[230301,230302],maximum_real_calls=2,budget_before=status()))
    rows=[]
    for seed in range(230201,230209):
        for option in (*OPTIONS,"learning"):
            env=MarketEnv(config);s=env.reset(episode_id=f"s23-common-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
            manifest=EpisodeManifest.create(env,s,cooperation_mode="combined_v1");tr=[]
            while not s.terminal:
                if s.state_version in (5,12):
                    env=MarketEnv(config);env.load_state(MarketState.from_dict(s.to_dict()))
                aa=actions(config,s);result=step(env,s,aa,None if option=="learning" else option)
                tr.append(MarketTransition.create(s,aa,result));s=result.state_after
            if seed==230201:
                verify_replay(MarketEnv(config),manifest,[MarketTransition.from_dict(t.to_dict()) for t in tr])
                write(f"{option}-replay.json",dict(manifest=manifest.to_dict(),transitions=[t.to_dict() for t in tr]))
            rows.append(dict(seed=seed,option=option,welfare_cents=s.welfare_accounting.cumulative_total_economic_welfare_cents,
                             consumer_surplus_cents=s.welfare_accounting.cumulative_consumer_surplus_cents,government_cash_cents=s.government.cash_cents))
            write("rows.json",rows)
    real=[]
    for seed in (230301,230302):
        env=MarketEnv(config);s=env.reset(episode_id=f"s23-real-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
        manifest=EpisodeManifest.create(env,s,cooperation_mode="combined_v1");tr=[]
        for _ in range(6):
            aa=actions(config,s);result=step(env,s,aa,None);tr.append(MarketTransition.create(s,aa,result));s=result.state_after
        try:
            d=await choose_option(actor_id="government",objective="Preserve fiscal solvency and improve remaining total welfare. Transfers alone do not create welfare. Choose from published programs; policy learning is observational.",
                observation=dict(round=s.round,remaining=s.rounds_remaining,own_cash_cents=s.government.cash_cents,
                    public_stockout=s.market.lost_after_stockout_orders,public_demand=s.market.realized_demand_orders,hhi=s.strategic_market.hhi_ppm,
                    published_memory=unpack(s.government.policy_memory)),options=[dict(id=o) for o in OPTIONS])
            aa=actions(config,s);alternatives={}
            for o in OPTIONS:
                shadow=MarketEnv(config);shadow.load_state(s);v=step(shadow,s,aa,o).state_after
                alternatives[o]=dict(welfare_cents=v.welfare_accounting.round_total_economic_welfare_cents,cash_cents=v.government.cash_cents)
            result=step(env,s,aa,d.option_id);tr.append(MarketTransition.create(s,aa,result))
            verify_replay(MarketEnv(config),manifest,[MarketTransition.from_dict(t.to_dict()) for t in tr])
            real.append(dict(seed=seed,choice=asdict(d),counterfactuals=alternatives,trace=tr[-1].to_dict(),passed=True))
        except Exception as exc:real.append(dict(seed=seed,passed=False,error=type(exc).__name__,message=str(exc),retry=False))
        write("real.json",real)
    summary=dict(episodes=len(rows),rounds=len(rows)*20,real_passed=all(x["passed"] for x in real),budget_after=status(),
        groups=[dict(option=o,n=8,mean_welfare_cents=mean(r["welfare_cents"] for r in rows if r["option"]==o),
                     mean_government_cash_cents=mean(r["government_cash_cents"] for r in rows if r["option"]==o)) for o in (*OPTIONS,"learning")],
        limits="Post-purchase rebates redistribute surplus, without household saving/dynamic demand. UCB explores realized welfare, not a guarantee of causal improvement or optimal long-horizon policy.")
    write("summary.json",summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":asyncio.run(main())
