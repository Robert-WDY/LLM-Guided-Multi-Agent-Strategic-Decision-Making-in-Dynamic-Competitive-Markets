"""Two-market prospective intervention study. Run from repository root."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/econevals_stage'), str(ROOT/'scripts'), str(ROOT/'src')]
from common import read, save
from game_theory_agent.market.protocols import state_hash
from duopoly import environment, settle, Opponent, ALPHAS
from reliable_v19_common import make, step, company_action, MarketEnv
from game_theory_agent.game_theory.cycle_pricing import decide, apply_native_price, HOLD, WARM
import numpy as np

OUT = ROOT/'experiment-results/pricing-feedback-v2'
OUT.mkdir(exist_ok=True, parents=True)
RAW = ROOT/'runs/pricing-feedback-v2'
RAW.mkdir(exist_ok=True, parents=True)
ARMS = ('rule','fixed','feedback','generic','conditional','wide','gated','freeze')
GATE_OFF = 1.e30


class World:
    """Evaluator owns the transition, intervention schedule and independent opponent tape."""
    def __init__(self, market, seed, change):
        self.market, self.seed, self.change = market, seed, change
        self.length = 60
        self.when = self.length//2 - 4 + seed%9
        self.up = seed%2 == 0
        self.t = 1
        if market == 'econ':
            self.kw,self.cost,_,_,self.high = environment(ALPHAS[seed%3])
            self.low, self.ref = self.high*.15, self.high*.54
            self.op = Opponent(seed, 'stationary',self.high)
        else:
            self.config,self.env,self.state = make(('normal','recession','tight_supply')[seed%3],2,seed,self.length)
            bounds = self.config.mapping('action','bounds')['price_cents']
            self.low,self.high = bounds['min'],bounds['max']
            self.ref = self.state.company('company_A').commercial.price_cents
        self.last = self.peer = self.ref

    def clone(self):
        tmp = object.__new__(World)
        tmp.__dict__.update(self.__dict__)
        if self.market == 'native':
            tmp.env = MarketEnv(self.config)
            tmp.env.load_state(self.state)
        return tmp

    def active(self):
        return self.market == 'econ' or 'company_A' in self.state.strategic_market.active_company_ids

    def legal(self, price):
        if self.market == 'econ':
            return price
        base = company_action(self.config,self.state,'company_A','balanced')
        if not self.active():
            return base.price_cents
        return apply_native_price(self.config,self.state,'company_A',base,price).price_cents

    def execute(self, price):
        change = self.change if self.t >= self.when else 'stable'
        if self.market == 'econ':
            target = self.op.price(self.t-1,self.last,self.peer)
            if change == 'rival':
                family = 'matching' if self.up else 'undercut'
                original = self.op.family
                self.op.family = family
                target = self.op.price(self.t-1,self.last,self.peer)
                self.op.family = original
            if change == 'reverse':
                target = np.clip(2*self.op.base-target-.65*(self.last-self.ref), self.low,self.high)
            kw = self.kw
            if change == 'demand':
                # A common shift in willingness to pay relative to the outside good.
                kw = dict(kw,a=((2.45 if self.up else 1.55),)*2)
            own = self.ref if price is None else price
            q,profits = settle(kw,self.cost,(own,target))
            profit,qty = profits[0],q[0]
            surplus = 100*kw['mu']*math.log(1+sum(math.exp((v-pr/alpha)/kw['mu']) for v,pr,alpha in zip(kw['a'],(own,target),kw['alpha'])))
        else:
            s = self.state
            if self.t == self.when and change == 'demand':
                s = replace(s,market=replace(s.market,base_demand_orders=round(s.market.base_demand_orders*(1.45 if self.up else .55))))
                s = replace(s,state_hash=state_hash(s.to_dict()))
                self.env.load_state(s)
            base = company_action(self.config,s,'company_A','balanced')
            own_action = base if price is None or not self.active() else apply_native_price(self.config,s,'company_A',base,price)
            rival = company_action(self.config,s,'company_B','balanced')
            if 'company_B' in s.strategic_market.active_company_ids:
                peer_price = self.ref*(1+.035*math.sin((self.seed+self.t)*.72))
                if change == 'rival':
                    peer_price = self.last*(1.04 if self.up else .91)
                if change == 'reverse':
                    peer_price = peer_price-.92*(self.last-self.ref)
                rival = apply_native_price(self.config,s,'company_B',rival,peer_price)
            self.state = step(self.env,s,{'company_A':own_action,'company_B':rival})
            firm = self.state.company('company_A')
            own,target = firm.commercial.price_cents,self.state.company('company_B').commercial.price_cents
            qty,profit = firm.commercial.sales_orders,firm.financial.round_profit_cents
            surplus = self.state.welfare_accounting.round_consumer_surplus_cents
        self.last,self.peer = float(own),float(target)
        self.t += 1
        return [self.last,self.peer,qty,profit,surplus]


def branch(world, price, steps):
    shadow = world.clone()
    return sum(.99**i*shadow.execute(price)[3] for i in range(steps))


def replay(market,seed,condition,arm,gate=GATE_OFF,labels=False,agent=None,selector=None):
    world = World(market,seed,condition)
    rows, contracts, probes = [], [], []
    prev = None
    frozen = None
    mode = {'learned':'conditional','diagnostic':'conditional','gated':'conditional','freeze':'feedback','rule':'feedback'}.get(arm,arm)
    for t in range(1,world.length+1):
        left = world.length-t+1
        new = prev is None or t > prev['end']
        original = None
        if agent and new and t>WARM and world.active():
            hint = decide(t,left,world.ref,world.low,world.high,rows,prev)['draft']
            original = agent(market,rows,world.low,world.high,hint,min(left,HOLD),left)
        dec = decide(t,left,world.ref,world.low,world.high,rows,prev,mode=mode,
                     gate=gate if arm=='gated' else 0,original=original)
        if new and t>WARM and arm in ('learned','diagnostic'):
            from selector import predict
            score=predict(selector,rows,dec)
            dec=dict(dec,selection_score=score,price=dec['proposed'] if score>gate else dec['draft'])
        if new:
            record = dict(dec,active=world.active())
            record['actual_override'] = (abs(dec['price']-dec['draft'])>1.e-8 and
                                         abs(world.legal(dec['price'])-world.legal(dec['draft']))>1.e-8)
            contracts.append(record)
            # Score every full opportunity on B's common path, including rejected proposals.
            if labels and t>WARM and left>=HOLD and world.active():
                proposed = decide(t,left,world.ref,world.low,world.high,rows,prev,mode='conditional',original=original)
                delta = branch(world,proposed['proposed'],HOLD)-branch(world,proposed['draft'],HOLD)
                probes.append(dict(t=t,score=proposed['score'],delta=delta,
                                   draft=proposed['draft'],proposed=proposed['proposed']))
        price = dec['price']
        if arm == 'freeze' and t>=world.when:
            if frozen is None:
                frozen = world.last
            price = frozen
        rows.append(world.execute(None if arm=='rule' else price))
        prev = dec
    return dict(seed=seed,market=market,condition=condition,arm=arm,when=world.when,
                rows=rows,contracts=contracts,labels=probes)


def worker(task):
    market,seed,phase,gate = task
    target = RAW/f'{phase}-{market}-{seed}.json.gz'
    conditions = ['stable','demand','rival'] + (['reverse'] if phase=='confirm' else [])
    arms = ARMS if phase!='cal' else ('feedback',)
    selector = None
    if phase=='confirm':
        arms = (*arms,'learned','diagnostic')
        spec=read(OUT/'learned-thresholds.json')[market]
        selector=read(OUT/'selectors.json')[market]
    def threshold(a):
        if a=='learned':return spec['threshold']
        if a=='diagnostic':return spec['diagnostic']
        return gate
    rows = [replay(market,seed,c,a,threshold(a),labels=a=='feedback',selector=selector) for c in conditions for a in arms]
    save(target,rows)
    return seed


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['dev','cal','confirm'])
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--market',choices=['econ','native','both'],default='both')
    args=p.parse_args()
    start,n={'dev':(51200,8),'cal':(52300,24),'confirm':(53500,48)}[args.phase]
    gates=read(OUT/'thresholds.json') if args.phase=='confirm' else {}
    tasks=[(m,s,args.phase,gates.get(m,{}).get('threshold',0)) for m in ('econ','native')
           if args.market in (m,'both') for s in range(start,start+n)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(worker,tasks))
    print(args.phase,len(tasks),flush=True)


if __name__ == '__main__':
    main()

