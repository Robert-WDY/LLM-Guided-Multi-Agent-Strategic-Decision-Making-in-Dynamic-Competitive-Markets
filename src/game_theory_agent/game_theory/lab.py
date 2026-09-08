"""Finite, auditable teaching games. These payoffs are not calibrated market data."""
from fractions import Fraction
import math
import random

GAMES = {
    "prisoners_dilemma": dict(title="囚徒困境", labels=["合作", "背叛"], payoffs=[[[3,3],[0,5]],[[5,0],[1,1]]], lesson="个体最佳回应可能偏离共同利益。"),
    "stag_hunt": dict(title="猎鹿博弈", labels=["猎鹿", "猎兔"], payoffs=[[[4,4],[0,3]],[[3,0],[3,3]]], lesson="两个纯均衡，合作需要对对方行动的信心。"),
    "chicken": dict(title="胆小鬼博弈", labels=["退让", "坚持"], payoffs=[[[3,3],[1,4]],[[4,1],[0,0]]], lesson="坚持的回报取决于对方是否退让。"),
    "matching_pennies": dict(title="匹配硬币", labels=["正面", "反面"], payoffs=[[[1,-1],[-1,1]],[[-1,1],[1,-1]]], lesson="没有纯均衡，随机化使对方无差异。"),
}

STRATEGIES={"cooperate":"始终合作","defect":"始终背叛","tit_for_tat":"针锋相对","grim_trigger":"冷酷触发","generous_tft":"宽容互惠","win_stay_lose_shift":"赢留输换"}


def _number(name,value,low,high,integer=False):
    if type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(f"{name} must be a {'whole' if integer else 'finite'} number between {low} and {high}")


def repeated_action(strategy,history,player,rng):
    if strategy not in STRATEGIES:raise ValueError("unknown repeated strategy")
    if strategy=="defect":return 1
    if strategy=="cooperate" or not history:return 0
    if strategy=="grim_trigger":return int(any(h['actions'][1-player]==1 for h in history))
    if strategy=="win_stay_lose_shift":
        last=history[-1];return last['actions'][player] if last['payoffs'][player]>=3 else 1-last['actions'][player]
    previous=history[-1]['actions'][1-player]
    return 0 if strategy=="generous_tft" and previous==1 and rng.random()<.2 else previous


def repeated(*,row="tit_for_tat",column="cooperate",rounds=50,noise=0.,discount=.95,seed=1):
    _number("rounds",rounds,1,500,integer=True)
    _number("noise",noise,0,0.5,integer=False)
    _number("discount",discount,0,1,integer=False)
    _number("seed",seed,0,2147483647,integer=True)
    if row not in STRATEGIES or column not in STRATEGIES:raise ValueError("unknown repeated strategy")
    if not 1<=rounds<=500 or not 0<=noise<=.5 or not 0<=discount<=1:raise ValueError("invalid repeated-game parameter")
    m=GAMES['prisoners_dilemma']['payoffs'];history=[]
    # Separate tremble and policy streams, so identical seeds give common execution noise.
    tremble=random.Random(seed);policies=[random.Random(seed+1000003),random.Random(seed+2000003)]
    for t in range(rounds):
        intended=[repeated_action(s,history,k,policies[k]) for k,s in enumerate((row,column))]
        flips=[tremble.random()<noise for _ in range(2)];actions=[a^int(f) for a,f in zip(intended,flips)]
        history.append(dict(round=t+1,intended=intended,actions=actions,trembles=flips,payoffs=m[actions[0]][actions[1]]))
    weights=[discount**t for t in range(rounds)];den=sum(weights)
    return dict(kind="repeated",row=row,column=column,seed=seed,rounds=rounds,noise=noise,discount=discount,history=history,
        average_payoffs=[sum(h['payoffs'][k] for h in history)/rounds for k in range(2)],
        discounted_payoffs=[sum(w*h['payoffs'][k] for w,h in zip(weights,history))/den for k in range(2)],
        cooperation_rate=[sum(h['actions'][k]==0 for h in history)/rounds for k in range(2)],
        infinite_grim_threshold=.5,scope="无限期完美观察下冷酷触发要求δ≥0.5；本次有限已知轮数，唯一阶段均衡为背叛，互惠轨迹不是子博弈精炼均衡证明。")


def learning(*,game="matching_pennies",algorithm="regret_matching",rounds=2000,seed=1):
    _number("rounds",rounds,10,10000,integer=True)
    _number("seed",seed,0,2147483647,integer=True)
    if game not in GAMES or algorithm not in ("regret_matching","fictitious_play") or not 10<=rounds<=10000:raise ValueError("invalid learning parameter")
    m=GAMES[game]['payoffs'];rng=random.Random(seed);regrets=[[0.,0.],[0.,0.]];counts=[[0,0],[0,0]];joint=[[0,0],[0,0]];history=[]
    for t in range(rounds):
        probs=[]
        for k in range(2):
            if algorithm=="regret_matching":
                positive=[max(0,r) for r in regrets[k]];probs.append(positive[0]/sum(positive) if sum(positive) else .5)
            else:
                other=counts[1-k];belief=(other[0]+1)/(sum(other)+2)
                values=[(expected(m,1-a,belief)[0] if k==0 else expected(m,belief,1-a)[1]) for a in range(2)]
                probs.append(1. if values[0]>values[1] else 0. if values[1]>values[0] else .5)
        i,j=[0 if rng.random()<p else 1 for p in probs];joint[i][j]+=1
        for k in range(2):
            for action in range(2):regrets[k][action]+=(m[action][j][0] if k==0 else m[i][action][1])-m[i][j][k]
            counts[k][(i,j)[k]]+=1
        if (t+1)%max(1,rounds//100)==0 or t==0 or t+1==rounds:
            history.append(dict(round=t+1,row_first_frequency=counts[0][0]/(t+1),column_first_frequency=counts[1][0]/(t+1),external_regret_per_round=[max(0,max(r))/(t+1) for r in regrets]))
    distribution=[[v/rounds for v in r] for r in joint]
    gains=[max(0.,max(sum(distribution[i][j]*((m[a][j][0] if k==0 else m[i][a][1])-m[i][j][k]) for i in range(2) for j in range(2)) for a in range(2))) for k in range(2)]
    return dict(kind="learning",game=game,algorithm=algorithm,rounds=rounds,seed=seed,history=history,joint_distribution=distribution,
        average_payoffs=[sum(distribution[i][j]*m[i][j][k] for i in range(2) for j in range(2)) for k in range(2)],
        cce_deviation_gains=gains,cce_gap=max(gains),marginal_nash_gains=exploitability(m,counts[0][0]/rounds,counts[1][0]/rounds),
        scope="外部遗憾检验经验联合分布的粗相关均衡缺口；边际乘积分布的纳什缺口单独计算。有限样本不保证单条轨迹收敛。")


def sequential(*,intercept=30,cost=6,maximum=24):
    _number("maximum",maximum,1,60,integer=True)
    _number("intercept",intercept,1,1000,integer=False)
    _number("cost",cost,0,999,integer=False)
    if not 1<=maximum<=60 or not 0<=cost<intercept<=1000:raise ValueError("invalid quantity-game parameter")
    def payoff(x,y):return (max(0,intercept-x-y)-cost)*x
    tree=[];pure=[]
    for x in range(maximum+1):
        best=max(payoff(y,x) for y in range(maximum+1));responses=[y for y in range(maximum+1) if payoff(y,x)==best]
        # Pessimistic follower tie breaking makes the commitment claim explicit.
        y=min(responses,key=lambda y:payoff(x,y));tree.append(dict(leader_quantity=x,follower_responses=responses,selected_follower=y,payoffs=[payoff(x,y),payoff(y,x)]))
        for y in responses:
            if payoff(x,y)==max(payoff(a,y) for a in range(maximum+1)):pure.append(dict(quantities=[x,y],payoffs=[payoff(x,y),payoff(y,x)]))
    best=max(r['payoffs'][0] for r in tree)
    return dict(kind="sequential",simultaneous_nash=pure,commitment_solutions=[r for r in tree if r['payoffs'][0]==best],tree=tree,
                scope="完全信息离散Cournot与先行者可承诺数量的Stackelberg；跟随者收益相同时采用对先行者最不利的回应。")


def information(*,weak_prior=.5,accuracy=.8,win=8,loss=6):
    _number("weak_prior",weak_prior,0,1,integer=False)
    _number("accuracy",accuracy,0.5,1,integer=False)
    _number("win",win,0,1000,integer=False)
    _number("loss",loss,0,1000,integer=False)
    if not 0<=weak_prior<=1 or not .5<=accuracy<=1 or not 0<win<=1000 or not 0<loss<=1000:raise ValueError("invalid information parameter")
    prior=weak_prior;baseline=max(0,prior*win-(1-prior)*loss);signals=[]
    for signal,likelihood in (("weak_signal",accuracy),("strong_signal",1-accuracy)):
        probability=prior*likelihood+(1-prior)*(1-likelihood)
        posterior=prior*likelihood/probability if probability else None
        value=posterior*win-(1-posterior)*loss if posterior is not None else None
        signals.append(dict(signal=signal,probability=probability,posterior_weak=posterior,entry_payoff=value,action="impossible" if value is None else "enter" if value>0 else "stay"))
    informed=sum(s['probability']*max(0,s['entry_payoff'] or 0) for s in signals)
    return dict(kind="information",weak_prior=prior,accuracy=accuracy,entry_threshold=loss/(win+loss),signals=signals,without_information=baseline,with_information=informed,value_of_information=max(0,informed-baseline),perfect_information_bound=prior*win-baseline,
        scope="两种外生对手类型、已知信号准确率；只做贝叶斯决策，不把外生信号称为内生信号博弈或完美贝叶斯均衡。")


def bargaining(*,surplus=100,disagreement_row=20,disagreement_column=10,weight=.5):
    _number("surplus",surplus,1,1000,integer=True)
    _number("disagreement_row",disagreement_row,0,1000,integer=False)
    _number("disagreement_column",disagreement_column,0,1000,integer=False)
    _number("weight",weight,0,1,integer=False)
    if not 1<=surplus<=1000 or not 0<=disagreement_row<=1000 or not 0<=disagreement_column<=1000 or not 0<weight<1:raise ValueError("invalid bargaining parameter")
    feasible=[dict(allocation=[x,surplus-x],gains=[x-disagreement_row,surplus-x-disagreement_column],score=(x-disagreement_row)**weight*(surplus-x-disagreement_column)**(1-weight)) for x in range(surplus+1) if x>=disagreement_row and surplus-x>=disagreement_column]
    best=max((r['score'] for r in feasible),default=None)
    return dict(kind="bargaining",surplus=surplus,disagreement=[disagreement_row,disagreement_column],weight=weight,solutions=[r for r in feasible if abs(r['score']-best)<1e-9],feasible=feasible,
                scope="可转移的固定剩余、整数分配、个体理性约束的加权Nash议价；是公理化解，不是已证明的谈判行为预测。")


def public_goods(*,players=4,endowment=10,contribution=4,multiplier=1.6):
    _number("players",players,2,8,integer=True)
    _number("endowment",endowment,0,100,integer=False)
    _number("contribution",contribution,0,100,integer=False)
    _number("multiplier",multiplier,1,8,integer=False)
    if not 2<=players<=8 or not 0<contribution<=endowment<=100 or not 1<multiplier<players:raise ValueError("public-good multiplier must be between 1 and player count")
    outcomes=[]
    for mask in range(2**players):
        actions=[(mask>>i)&1 for i in range(players)];total=sum(actions)*contribution;benefit=multiplier*total/players
        payoffs=[endowment-a*contribution+benefit for a in actions]
        outcomes.append(dict(actions=actions,payoffs=payoffs,total_payoff=sum(payoffs),pure_nash=not any(actions)))
    return dict(kind="public_goods",players=players,outcomes=outcomes,individual_marginal_return=contribution*(multiplier/players-1),social_marginal_return=contribution*(multiplier-1),
                scope="线性公共品：贡献有真实资源成本且收益均分；个体净回报为负、集体净回报为正，不把它直接套到当前市场参数。")


def matrix(value):
    if not isinstance(value,list) or len(value)!=2 or any(not isinstance(row,list) or len(row)!=2 for row in value):
        raise ValueError("payoffs must be a 2 by 2 matrix")
    for row in value:
        for cell in row:
            if not isinstance(cell,list) or len(cell)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e9 for v in cell):
                raise ValueError("each cell needs two finite payoffs within 1e9")
    return value


def expected(m,p,q):
    return [sum((p if i==0 else 1-p)*(q if j==0 else 1-q)*m[i][j][k] for i in range(2) for j in range(2)) for k in range(2)]


def exploitability(m,p,q):
    u=expected(m,p,q)
    return [max(expected(m,a,q)[0] for a in (0,1))-u[0],max(expected(m,p,b)[1] for b in (0,1))-u[1]]


def analyze(value):
    m=matrix(value);cells=[];pure=[];frontier=[]
    for i in range(2):
        for j in range(2):
            gains=[max(m[x][j][0] for x in range(2))-m[i][j][0],max(m[i][y][1] for y in range(2))-m[i][j][1]]
            pareto=not any(all(m[x][y][k]>=m[i][j][k] for k in range(2)) and any(m[x][y][k]>m[i][j][k] for k in range(2)) for x in range(2) for y in range(2))
            cells.append(dict(row=i,column=j,payoffs=m[i][j],deviation_gains=gains,best_responses=[g==0 for g in gains],pareto=pareto))
            if gains==[0,0]:pure.append([i,j])
            if pareto:frontier.append([i,j])
    # Fractions preserve exact decimal payoff comparisons in the 2x2 formula.
    a,b,c,d=[Fraction(str(m[i][j][0])) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    e,f,g,h=[Fraction(str(m[i][j][1])) for i,j in ((0,0),(0,1),(1,0),(1,1))]
    mixed=[];den_a=a-b-c+d;den_b=e-f-g+h
    if den_a and den_b:
        q=(d-b)/den_a;p=(h-g)/den_b
        if 0<p<1 and 0<q<1:
            mixed.append(dict(row_probability=float(p),column_probability=float(q),exact=[str(p),str(q)],payoffs=expected(m,float(p),float(q)),deviation_gains=exploitability(m,float(p),float(q))))
    dominated=[]
    for k in range(2):
        for action in range(2):
            if all((m[action][o][0]<m[1-action][o][0] if k==0 else m[o][action][1]<m[o][1-action][1]) for o in range(2)):
                dominated.append(dict(player=k,action=action,by=1-action))
    security=[]
    for k in range(2):
        u=[[m[i][j][k] if k==0 else m[j][i][k] for j in range(2)] for i in range(2)]
        candidates=[0.,1.];den=u[0][0]-u[1][0]-u[0][1]+u[1][1]
        if den:
            p=(u[1][1]-u[1][0])/den
            if 0<=p<=1:candidates.append(p)
        score=lambda p:min(p*u[0][o]+(1-p)*u[1][o] for o in range(2))
        best=max(candidates,key=score);security.append(dict(player=k,first_probability=best,guaranteed_payoff=score(best)))
    return dict(payoffs=m,cells=cells,pure_nash=pure,interior_mixed_nash=mixed,strictly_dominated=dominated,pareto_cells=frontier,security=security,
                degenerate=not bool(den_a and den_b),scope="固定2×2同时行动；展示所有纯均衡及非退化内部混合均衡，退化连续均衡不穷尽；帕累托仅比较四个纯动作结果，不包含随机化组合。")
