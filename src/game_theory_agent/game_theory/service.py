"""Local controller-only concept experiments, immutable JSON persistence and export."""
import hashlib
import json
import threading
from datetime import datetime,UTC
from pathlib import Path
from typing import Any,Literal
from fastapi import APIRouter,Header,HTTPException,Query
from pydantic import BaseModel,ConfigDict,Field
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.actor_experiments import write_json,read_json
from . import lab
from .market_strategies import MODES,market_matrix

KINDS={"advisor":"目标驱动博弈建议","matrix":"收益矩阵与均衡","repeated":"重复博弈与互惠","learning":"策略学习与遗憾","sequential":"序贯行动与承诺","information":"贝叶斯信息价值","bargaining":"议价与分歧点","public_goods":"公共品与搭便车","market_matrix":"当前市场反事实"}


class ExperimentRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:str=Field(min_length=1,max_length=100)
    kind:Literal['matrix','repeated','learning','sequential','information','bargaining','public_goods','market_matrix','advisor']
    parameters:dict[str,Any]=Field(default_factory=dict)


def checksum(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class TheoryService:
    def __init__(self,root,config,state_provider=None,history_provider=None):
        self.root=Path(root);self.config=config;self.state_provider=state_provider;self.history_provider=history_provider;self.lock=threading.RLock()

    def execute(self,request):
        key=hashlib.sha256(request.request_id.encode()).hexdigest();path=self.root/(key+'.json')
        payload=request.model_dump();identity=checksum(dict(request=payload,config_hash=self.config.config_sha256))
        with self.lock:
            if path.exists():
                old=self.get(key)
                if old['identity']!=identity:raise ValueError('request id already belongs to different inputs')
                return old
            p=dict(request.parameters)
            if request.kind=='advisor':
                from .objectives import AdviceRequest
                from .advisor_search import advise
                version=p.pop('advisor_version','v16')
                if version=='v17':
                    from .advisor_beliefs import RobustRequest as AdviceRequest
                    from .advisor_robust import advise
                elif version=='reliable':
                    from .reliable_advisor import ReliableRequest as AdviceRequest
                    from .reliable_advisor import advise
                elif version!='v16':raise ValueError('unknown advisor version')
                episode=p.pop('episode_id',None);count=p.pop('company_count',4);rounds=p.pop('market_rounds',10)
                if type(count)!=int or not 2<=count<=10:raise ValueError('company_count must be between 2 and 10')
                if type(rounds)!=int or rounds not in (5,10,15,20,60):raise ValueError('unsupported market_rounds')
                spec=AdviceRequest.model_validate(p)
                if episode:
                    if not self.state_provider:raise ValueError('no current market session')
                    state=self.state_provider(episode)
                    if version=='v17' and self.history_provider:spec=AdviceRequest.model_validate({**spec.model_dump(),'history':self.history_provider(episode,state.round)})
                    if version=='reliable' and self.history_provider:spec=AdviceRequest.model_validate({**spec.model_dump(),'public_history':self.history_provider(episode,state.round)})
                else:state=MarketEnv(self.config).reset(company_ids=[f'company_{chr(65+i)}' for i in range(count)],episode_id=f'advisor-{spec.seed}',episode_seed=spec.seed,max_rounds=rounds,cooperation_mode='combined_v1')
                result=advise(self.config,state,spec)
            elif request.kind=='matrix':
                game=p.pop('game','prisoners_dilemma');payoffs=p.pop('payoffs',None)
                if game not in lab.GAMES or p:raise ValueError('unknown matrix inputs')
                result=dict(kind='matrix',game=game,labels=lab.GAMES[game]['labels'],**lab.analyze(payoffs if payoffs is not None else lab.GAMES[game]['payoffs']))
            elif request.kind=='market_matrix':
                episode=p.pop('episode_id',None);seed=p.pop('seed',330101)
                if type(seed)!=int or not 0<=seed<=2**31-1:raise ValueError('invalid market seed')
                if episode:
                    if not self.state_provider:raise ValueError('no current market session')
                    state=self.state_provider(episode)
                else:state=MarketEnv(self.config).reset(episode_id=f'theory-matrix-{seed}',episode_seed=seed,max_rounds=10)
                result=dict(kind='market_matrix',source_round=state.round,**market_matrix(self.config,state,**p))
            else:result=getattr(lab,request.kind)(**p)
            value=dict(id=key,identity=identity,created=datetime.now(UTC).isoformat(),request=payload,config_hash=self.config.config_sha256,result=result)
            value['sha256']=checksum(value);write_json(path,value);return value

    def get(self,key):
        if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('invalid experiment id')
        value=read_json(self.root/(key+'.json'))
        if not isinstance(value,dict) or not all(k in value for k in ('sha256','id','created','request','result','identity')):
            raise ValueError('invalid experiment document')
        check=dict(value);digest=check.pop('sha256')
        if checksum(check)!=digest:raise ValueError('experiment checksum mismatch')
        return value

    def listing(self,offset=0,limit=30):
        if type(offset)!=int or offset<0 or type(limit)!=int or not 1<=limit<=100:
            raise ValueError('invalid history page')
        paths=sorted(self.root.glob('*.json'),key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)
        rows=[];issues=[]
        for p in paths[offset:offset+limit]:
            try:
                v=self.get(p.stem);rows.append({k:v[k] for k in ('id','created','request','sha256')})
            except (ValueError,TypeError,KeyError,OSError):
                issues.append(dict(id=p.stem,message='记录损坏或不可读，原文件已保留；其他实验可继续读取。'))
        end=offset+limit
        return dict(experiments=rows,issues=issues,total=len(paths),next_offset=end if end<len(paths) else None)


def router(root,config,require_token,state_provider=None,history_provider=None):
    routes=APIRouter(prefix='/api/v1/controller/theory-lab');service=TheoryService(root,config,state_provider,history_provider)
    @routes.get('/catalog')
    def catalog(token:str|None=Header(default=None,alias='X-Controller-Token')):
        require_token(token)
        return dict(kinds=KINDS,games=lab.GAMES,strategies=lab.STRATEGIES,market_modes=MODES)
    @routes.post('/experiments')
    def execute(request:ExperimentRequest,token:str|None=Header(default=None,alias='X-Controller-Token')):
        require_token(token)
        try:return service.execute(request)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:raise HTTPException(422,str(exc)) from exc
    @routes.get('/experiments')
    def listing(token:str|None=Header(default=None,alias='X-Controller-Token'),offset:int=Query(default=0,ge=0),limit:int=Query(default=30,ge=1,le=100)):
        require_token(token);return service.listing(offset,limit)
    @routes.get('/experiments/{key}')
    def get(key:str,token:str|None=Header(default=None,alias='X-Controller-Token')):
        require_token(token)
        try:return service.get(key)
        except FileNotFoundError as exc:raise HTTPException(404,'experiment not found') from exc
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    @routes.get('/acceptance')
    def acceptance(token:str|None=Header(default=None,alias='X-Controller-Token')):
        require_token(token);base=Path(__file__).resolve().parents[3]/'runs';reports={}
        for stage in range(29,35):
            p=base/f'theory-stage{stage}'/'summary.json'
            if p.exists():reports[str(stage)]={k:v for k,v in read_json(p).items() if k not in ('results','history')}
        audit=base/'advisor-v16'
        for stage in range(1,7):
            p=audit/f'stage{stage}.json'
            if p.exists():reports[f'v16-{stage}']={k:v for k,v in read_json(p).items() if k not in ('results','traces')}
        for stage in range(1,7):
            p=base/'advisor-v17'/f'stage{stage}.json'
            if p.exists():reports[f'v17-{stage}']={k:v for k,v in read_json(p).items() if k not in ('results','traces')}
        p=base/'reliable-advisor-v19'/'acceptance.json'
        if p.exists():reports['reliable']=read_json(p)
        return reports
    return routes
