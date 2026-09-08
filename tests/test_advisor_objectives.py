from game_theory_agent.game_theory.objectives import AdviceRequest,objective,aggregate
from game_theory_agent.market import load_market_config,MarketEnv
from pathlib import Path
import pytest
CONFIG=load_market_config(Path(__file__).parents[1]/'configs/market_v14_local.yaml')
def test_objective_tradeoffs_and_hard_constraint():
 s=MarketEnv(CONFIG).reset();a=dict(profit=3000000,welfare=-1000000,cash=3000000,minimum_cash=100,exited=False);b=dict(profit=1000000,welfare=5000000,cash=1000000,minimum_cash=100,exited=False)
 profit=objective(AdviceRequest(goal='profit'),s);welfare=objective(AdviceRequest(goal='welfare'),s)
 assert aggregate([a,a],profit)['score']>aggregate([b,b],profit)['score']
 assert aggregate([a,a],welfare)['score']<aggregate([b,b],welfare)['score']
 assert not aggregate([dict(a,minimum_cash=-1)],profit)['safe']
 assert sum(profit['weights'].values())==1
 for phase in ('survival','expansion','mature','catch_up','closing'):
  spec=objective(AdviceRequest(phase=phase),s);assert spec['phase']==phase and abs(sum(spec['weights'].values())-1)<1e-9
 with pytest.raises(ValueError):AdviceRequest(goal='custom',weights={'profit':.5})
 with pytest.raises(ValueError):AdviceRequest(horizon=True)
