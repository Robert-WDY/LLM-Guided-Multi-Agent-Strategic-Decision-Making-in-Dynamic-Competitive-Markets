"""Executable company policies with public/own information and inspectable decisions."""
from copy import deepcopy
from statistics import mean
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.actor_policies import COMPANY_OPTIONS,company_action,observation
from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.strategic_reliability.public_rollout import build_public_forecast_state

MODES={"theory_best_response":"公开预测最佳回应","theory_maximin":"公开预测保底策略","theory_tft":"公开价格互惠","theory_grim":"价格触发持续竞争","theory_wsls":"收益赢留输换"}


def choose(config,state,cid,mode,previous):
    if mode not in MODES or cid not in state.company_ids:raise ValueError("game-theory policies support companies only")
    observed=observation(state,cid);memory=deepcopy(previous or {})
    prices={k:v for k,v in observed['prices'].items() if k!=cid};old=memory.get('prices',prices)
    movement=mean(prices[k]/max(1,old.get(k,prices[k]))-1 for k in prices) if prices else 0
    signal='value' if movement<-.01 else 'margin' if movement>.01 else 'balanced'
    counts=memory.setdefault('counts',{'value':1,'balanced':1,'margin':1})
    if 'prices' in memory:counts[signal]+=1
    memory['triggered']=memory.get('triggered',False) or signal=='value'
    detail=dict(mode=mode,signal=signal,public_price_change=movement,uses_hidden_state=False)
    if mode=='theory_tft':option='value' if signal=='value' else 'balanced'
    elif mode=='theory_grim':option='value' if memory['triggered'] else 'balanced'
    elif mode=='theory_wsls':
        option=memory.get('option','balanced')
        if 'profit' in memory and observed['profit']<memory['profit']:option='margin' if option=='balanced' else 'balanced'
    else:
        legal=ObservationBuilder().build(state,cid,'public')
        forecast,record=build_public_forecast_state(config=config,observation=legal,company_id=cid,persona_profile=PersonaRegistry.from_market_config(config).get('balanced_v1'))
        scenario_options=('value','balanced','margin');table=[]
        for own in COMPANY_OPTIONS:
            payoffs=[]
            for opponent in scenario_options:
                env=MarketEnv(config);env.load_state(forecast)
                actions={c:company_action(config,forecast,c,own if c==cid else opponent) for c in forecast.company_ids}
                result=env.step(f'{forecast.episode_id}:{forecast.round}:{forecast.state_version}',actions)
                payoffs.append(result.state_after.company(cid).financial.round_profit_cents)
            table.append(dict(option=own,scenario_profits_cents=payoffs,expected_profit_cents=sum(p*counts[s] for p,s in zip(payoffs,scenario_options))/sum(counts.values()),worst_profit_cents=min(payoffs)))
        metric='expected_profit_cents' if mode=='theory_best_response' else 'worst_profit_cents'
        best=max(table,key=lambda r:r[metric]);option=best['option']
        detail.update(forecast=record.model_dump(mode='json'),scenario_options=list(scenario_options),belief_counts=counts.copy(),payoff_table=table,criterion=metric,
                      scope='单轮、四个自身选项、三种全体对手情景的公开预测；不是真实市场全信息收益或动态纳什均衡。')
    memory.update(prices=prices,profit=observed['profit'],option=option)
    detail['selected_option']=option
    return option,detail,memory


def market_matrix(config,state,actor='company_A',opponent='company_B'):
    """Research-only exact fixed-state counterfactual, never passed into actor prompts."""
    if actor==opponent or actor not in state.company_ids or opponent not in state.company_ids or state.terminal:raise ValueError('two distinct live company IDs and a nonterminal state are required')
    choices=('balanced','value');payoffs=[];hashes=[]
    for own in choices:
        row=[]
        for other in choices:
            env=MarketEnv(config);env.load_state(state)
            actions={c:company_action(config,state,c,own if c==actor else other if c==opponent else 'balanced') for c in state.company_ids}
            result=env.step(f'{state.episode_id}:{state.round}:{state.state_version}',actions).state_after
            row.append([result.company(actor).financial.round_profit_cents,result.company(opponent).financial.round_profit_cents]);hashes.append(result.state_hash)
        payoffs.append(row)
    from .lab import analyze
    return dict(**analyze(payoffs),actors=[actor,opponent],labels=list(choices),source_state_hash=state.state_hash,outcome_hashes=hashes,
                information_scope='研究者完整状态单轮反事实，其他企业保持balanced；同一初始状态与随机种子。不是公司的合法预测输入，也不是整个动态市场均衡。')
