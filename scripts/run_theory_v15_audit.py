"""Free, preregistered v15 gap acceptance; no model client or budget mutation."""
import asyncio
import hashlib
import json
from pathlib import Path
from game_theory_agent.game_theory.service import TheoryService,ExperimentRequest
from game_theory_agent.game_theory.lab import GAMES
from game_theory_agent.game_theory.market_strategies import MODES
from game_theory_agent.market import load_market_config
from game_theory_agent.market.actor_experiments import run_episode,write_json,read_json
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'runs/theory-v15-audit'
async def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'summary.json').exists():print(json.dumps(read_json(DEST/'summary.json')));return
    config=load_market_config(ROOT/'configs/market_v14_local.yaml')
    before=status();old=ROOT/'releases/local-market-v15.0.0/local-market-v15.0.0-source.zip'
    old_hash=hashlib.sha256(old.read_bytes()).hexdigest()
    spec=dict(games=list(GAMES),algorithms=['regret_matching','fictitious_play'],rounds=[201,2999,3001],seeds=[350101,350102],market_seed=330101,market_rounds=20,market_modes=['rule',*MODES],new_real_calls=0)
    write_json(DEST/'preregistration.json',spec)
    service=TheoryService(ROOT/'.local-state-v14-release/workbench/theory-lab',config)
    rows=[]
    for game in spec['games']:
      for algorithm in spec['algorithms']:
       for rounds in spec['rounds']:
        for seed in spec['seeds']:
         request=ExperimentRequest(request_id=f'v15-audit-{game}-{algorithm}-{rounds}-{seed}',kind='learning',parameters=dict(game=game,algorithm=algorithm,rounds=rounds,seed=seed))
         record=service.execute(request);r=record['result'];last=r['history'][-1]
         assert last['round']==rounds
         assert max(abs(x-y) for x,y in zip(last['external_regret_per_round'],r['cce_deviation_gains']))<1e-10
         assert abs(last['row_first_frequency']-sum(r['joint_distribution'][0]))<1e-10
         assert abs(last['column_first_frequency']-sum(row[0] for row in r['joint_distribution']))<1e-10
         rows.append(dict(id=record['id'],**request.parameters,last_round=last['round'],cce_gap=r['cce_gap'],sha256=record['sha256']))
    write_json(DEST/'learning.json',rows)
    market=[];historical=read_json(ROOT/'runs/theory-stage33/summary.json')['results']
    for mode in spec['market_modes']:
      result=await run_episode(config,directory=DEST/f'market-{mode}',seed=spec['market_seed'],rounds=20,modes={'company_A':mode})
      original=next(r for r in historical if r['variant']==mode and r['seed']==spec['market_seed'])
      assert result['replay_passed'] and result['state_hash']==original['state_hash']
      market.append(dict(mode=mode,state_hash=result['state_hash'],matches_v15=True,replay_passed=True))
    real=read_json(ROOT/'runs/theory-stage34/summary.json');verified=[]
    for decision in real['decisions']:
      record=read_json(ROOT/'runs/theory-stage34'/f"{decision['case']}.json")
      option=record['response']['option_id'];scores=record['case']['scores'];loss=max(scores.values())-scores[option]
      assert record['status']=='complete' and loss==record['deviation_loss']==0
      verified.append(dict(case=decision['case'],option=option,loss=loss))
    assert before==status() and hashlib.sha256(old.read_bytes()).hexdigest()==old_hash
    report=dict(passed=True,learning_runs=len(rows),learning_rounds=sum(r['rounds'] for r in rows),market_episodes=len(market),market_rounds=120,market=market,historical_real_records_verified=verified,new_real_calls=0,budget_unchanged=before,old_v15_sha256=old_hash,conclusion='Final-round traces match certificates at non-grid horizons; six market replays match frozen evidence exactly. No new strategy superiority claim.')
    write_json(DEST/'summary.json',report);print(json.dumps(report,ensure_ascii=True))
if __name__=='__main__':asyncio.run(main())
