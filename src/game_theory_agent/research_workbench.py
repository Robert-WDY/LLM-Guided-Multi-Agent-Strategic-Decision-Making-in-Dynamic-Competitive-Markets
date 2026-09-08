"""Local durable serial experiment queue; all paid calls share the global ledger."""
import asyncio
import json
import sqlite3
import threading
import uuid
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import mean,stdev
from fastapi import APIRouter,Header,HTTPException
from pydantic import BaseModel,ConfigDict,Field
from .market import MarketConfig,MarketEnv
from .market.actor_policies import actor_ids,options
from .market.actor_experiments import run_episode,read_json,authorize_schema_retry
from .game_theory.market_strategies import MODES as THEORY_MODES
from .game_theory.objectives import ADVISOR_MODES
from .game_theory.advisor_robust import ROBUST_MODES

PARAMETERS={
 "supplier_cash":("supply_chain","transaction_accounting","initial_supplier_cash_cents"),
 "inventory_target":("supply_chain","strategic_policy","inventory_target_ppm"),
 "government_program":("autonomous_market","government","strategic_policy","max_program_cents"),
 "government_cash":("autonomous_market","government","initial_cash_cents"),
}


def active_jobs(root):
    database=Path(root)/"queue.sqlite3"
    if not database.exists():return 0
    with sqlite3.connect(database,timeout=5) as db:
        return db.execute("SELECT count(*) FROM jobs WHERE status IN ('running','queued')").fetchone()[0]


class Variant(BaseModel):
    model_config=ConfigDict(extra="forbid")
    label:str=Field(min_length=1,max_length=60)
    modes:dict[str,str]=Field(default_factory=dict)
    fixed:dict[str,str]=Field(default_factory=dict)
    overrides:dict[str,int]=Field(default_factory=dict)


class BatchRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    request_id:str=Field(min_length=1,max_length=80)
    seeds:list[int]=Field(min_length=1,max_length=16)
    rounds:int=Field(default=10,ge=5,le=60)
    company_count:int=Field(default=4,ge=2,le=10,strict=True)
    variants:list[Variant]=Field(min_length=1,max_length=8)
    authorize_real:bool=False
    model_name:str="deepseek-v4-flash"


def configured(config,variant):
    raw=config.to_dict()
    for key,value in variant["overrides"].items():
        if key not in PARAMETERS or value<0 or value>10**12:raise ValueError("unsupported experiment parameter")
        target=raw
        for part in PARAMETERS[key][:-1]:target=target[part]
        target[PARAMETERS[key][-1]]=value
    result=MarketConfig.from_mapping(raw)
    if not result.data.get("four_actor_policies"):raise ValueError("four-actor experiments require the complete v14 configuration")
    return result


def statistics(rows):
    groups=[]
    for label in dict.fromkeys(r["variant"] for r in rows):
        data=[r for r in rows if r["variant"]==label]
        w=[r["welfare_cents"] for r in data]
        groups.append(dict(variant=label,n=len(w),mean_welfare_cents=mean(w),std_welfare_cents=stdev(w) if len(w)>1 else None,
                           mean_company_profit_cents=mean(r["company_profit_cents"] for r in data),
                           mean_supplier_profit_cents=mean(r["supplier_profit_cents"] for r in data),
                           mean_consumer_surplus_cents=mean(r["consumer_surplus_cents"] for r in data)))
    comparisons=[]
    if groups:
        baseline={r["seed"]:r for r in rows if r["variant"]==groups[0]["variant"]}
        for g in groups[1:]:
            paired=[r["welfare_cents"]-baseline[r["seed"]]["welfare_cents"] for r in rows if r["variant"]==g["variant"] and r["seed"] in baseline]
            comparisons.append(dict(variant=g["variant"],baseline=groups[0]["variant"],paired_n=len(paired),
                mean_delta_cents=mean(paired) if paired else None,wins=sum(d>0 for d in paired),ties=sum(d==0 for d in paired)))
    return dict(groups=groups,paired_comparisons=comparisons,interpretation="Paired synthetic-seed comparisons. Small samples and co-adaptation do not establish real-world superiority.")


class Workbench:
    def __init__(self,root,config,*,start_worker=True):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.config=config
        self.database=self.root/"queue.sqlite3";self.wake=threading.Event();self.stopping=False;self.start_worker=start_worker
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_id TEXT UNIQUE, payload TEXT, config TEXT, status TEXT, progress TEXT, results TEXT, error TEXT, cancel INTEGER DEFAULT 0, created TEXT DEFAULT CURRENT_TIMESTAMP)")
            db.execute("UPDATE jobs SET status='interrupted',error='服务中断；可恢复已落盘回合，未知付费调用不会自动重试' WHERE status='running'")
        if start_worker:
            threading.Thread(target=self.worker,daemon=True,name="market-research-queue").start();self.wake.set()

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.database,timeout=30);db.row_factory=sqlite3.Row;db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:yield db
        finally:db.close()

    def submit(self,request):
        payload=request.model_dump()
        if payload['company_count']==4:payload.pop('company_count')
        keys={"deepseek-v4-flash":"DEEPSEEK_API_KEY","doubao-seed-2-0-lite-260215":"ARK_API_KEY"}
        if request.model_name not in keys:raise ValueError("unsupported budgeted model")
        if any("model" in v.modes.values() for v in request.variants) and request.authorize_real and not os.getenv(keys[request.model_name]):
            raise ValueError("所选供应商未配置API凭据；未发起付费调用")
        if len(payload["seeds"])*len(payload["variants"])>64 or len(set(payload["seeds"]))!=len(payload["seeds"]) or any(s<0 or s>2**31-1 for s in payload["seeds"]):
            raise ValueError("use distinct nonnegative seeds and at most 64 cases")
        if len({v["label"] for v in payload["variants"]})!=len(payload["variants"]):raise ValueError("variant labels must be unique")
        for variant in payload["variants"]:
            config=configured(self.config,variant);e=MarketEnv(config);ids=actor_ids(e.reset(company_ids=[f'company_{chr(65+i)}' for i in range(request.company_count)],episode_id="queue-validation",max_rounds=request.rounds))
            if set(variant["modes"])-set(ids) or set(variant["fixed"])-set(ids):raise ValueError("unknown actor")
            if any(m not in ("rule","learning","model","neural",*THEORY_MODES,*ADVISOR_MODES,*ROBUST_MODES) for m in variant["modes"].values()):raise ValueError("unknown controller")
            if any(m in (*THEORY_MODES,*ADVISOR_MODES,*ROBUST_MODES) and not a.startswith('company_') for a,m in variant['modes'].items()):raise ValueError("game-theory policies support companies only")
            if any(o not in options(a) for a,o in variant["fixed"].items()):raise ValueError("unknown portfolio")
            if "model" in variant["modes"].values() and not request.authorize_real:raise ValueError("real model authorization is required")
            if "neural" in variant["modes"].values():
                from .market.neural_policy import predict
                predict("company_A",{},config=config)
        with self.connect() as db:
            previous=db.execute("SELECT id,payload FROM jobs WHERE request_id=?",(request.request_id,)).fetchone()
            if previous:
                if json.loads(previous["payload"])!=payload:raise ValueError("request id reused with different inputs")
                return self.get(previous["id"])
            job_id=uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,request_id,payload,config,status,progress,results) VALUES(?,?,?,?,?,?,?)",
                (job_id,request.request_id,json.dumps(payload),json.dumps(self.config.to_dict()),"queued","{}","[]"))
        self.wake.set();return self.get(job_id)

    def get(self,job_id):
        with self.connect() as db:r=db.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
        if not r:raise KeyError("experiment job not found")
        value=dict(r)
        for k in ("payload","config","progress","results"):value[k]=json.loads(value[k])
        value["statistics"]=statistics(value["results"]);return value

    def listing(self):
        with self.connect() as db:ids=[r[0] for r in db.execute("SELECT id FROM jobs ORDER BY created DESC,rowid DESC LIMIT 100")]
        return [{k:v for k,v in self.get(i).items() if k!="config"} for i in ids]

    def cancelled(self,job_id):
        with self.connect() as db:return bool(db.execute("SELECT cancel FROM jobs WHERE id=?",(job_id,)).fetchone()[0])

    def control(self,job_id,action):
        job=self.get(job_id)
        if action=="rebuild-free":
            if job["status"]!="failed" or any("model" in v["modes"].values() for v in job["payload"]["variants"]):
                raise ValueError("only failed batches without any real model calls can be rebuilt")
            source=(self.root/job_id).resolve()
            if source.parent!=self.root.resolve() or source.is_symlink():raise ValueError("invalid experiment directory")
            archive=self.root/"recovery-evidence"/(job_id+"-"+uuid.uuid4().hex)
            archive.parent.mkdir(exist_ok=True)
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT status FROM jobs WHERE id=?",(job_id,)).fetchone()[0]!="failed":raise ValueError("job state changed")
                if source.exists():source.rename(archive)
                db.execute("UPDATE jobs SET cancel=0,status='queued',error=NULL,progress='{}',results='[]' WHERE id=?",(job_id,))
            self.wake.set();return self.get(job_id)
        if action=="retry-schema":
            if job["status"]!="failed":raise ValueError("only a failed job can be reviewed")
            found=False
            for p in (self.root/job_id).glob("case-*/round-*.json"):
                journal=read_json(p)
                for actor,d in journal["decisions"].items():
                    if d["status"]=="failed_schema":
                        authorize_schema_retry(p.parent,int(p.stem.split("-")[1]),actor,"Owner explicitly requested one schema retry; prior reservation retained")
                        found=True
            if not found:raise ValueError("No completed schema failure; unknown network outcomes cannot be retried")
            action="resume"
        with self.connect() as db:
            if action=="cancel":
                db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?",(job_id,))
            elif action=="resume":
                if job["status"] not in ("interrupted","cancelled","failed"):raise ValueError("only interrupted, cancelled or failed jobs can resume")
                db.execute("UPDATE jobs SET cancel=0,status='queued',error=NULL WHERE id=?",(job_id,))
            else:raise ValueError("unknown queue action")
        self.wake.set();return self.get(job_id)

    def worker(self):
        while not self.stopping:
            self.wake.wait(1);self.wake.clear()
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row=db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
                if row:db.execute("UPDATE jobs SET status='running' WHERE id=?",(row[0],))
            if row:self.execute(row[0]);self.wake.set()

    def execute(self,job_id):
        job=self.get(job_id);p=job["payload"];rows=list(job["results"]);base=MarketConfig.from_mapping(job["config"])
        try:
            for vi,variant in enumerate(p["variants"]):
                config=configured(base,variant)
                for seed in p["seeds"]:
                    if any(r["variant"]==variant["label"] and r["seed"]==seed for r in rows):continue
                    def progress(n,total):
                        with self.connect() as db:db.execute("UPDATE jobs SET progress=? WHERE id=?",
                            (json.dumps(dict(variant=variant["label"],seed=seed,round=n,rounds=total,completed_cases=len(rows),total_cases=len(p["variants"])*len(p["seeds"]))),job_id))
                    r=asyncio.run(run_episode(config,directory=self.root/job_id/f"case-{vi}-{seed}",seed=seed,rounds=p["rounds"],
                        modes=variant["modes"],fixed=variant["fixed"],authorize_real=p["authorize_real"],model_name=p["model_name"],company_count=p.get('company_count',4),
                        on_progress=progress,cancelled=lambda:self.cancelled(job_id)))
                    if r["status"]=="cancelled":
                        with self.connect() as db:db.execute("UPDATE jobs SET status='cancelled' WHERE id=?",(job_id,))
                        return
                    rows.append({k:v for k,v in r.items() if k!="memory"}|dict(variant=variant["label"],case=f"case-{vi}-{seed}"))
                    with self.connect() as db:db.execute("UPDATE jobs SET results=? WHERE id=?",(json.dumps(rows),job_id))
            with self.connect() as db:db.execute("UPDATE jobs SET status='complete',error=NULL,progress=? WHERE id=?",
                (json.dumps(dict(completed_cases=len(rows),total_cases=len(p["variants"])*len(p["seeds"]),round=p["rounds"],rounds=p["rounds"])),job_id))
        except Exception as exc:
            with self.connect() as db:db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",(f"{type(exc).__name__}: {exc}",job_id))

    def export(self,job_id):
        job=self.get(job_id);cases={}
        for vi,v in enumerate(job["payload"]["variants"]):
            for seed in job["payload"]["seeds"]:
                path=self.root/job_id/f"case-{vi}-{seed}"
                if (path/"checkpoint.json").exists():
                    cases[path.name]=dict(checkpoint=read_json(path/"checkpoint.json"),
                        decisions=[read_json(p) for p in sorted(path.glob("round-*.json"))])
        return dict(job=job,cases=cases,contains_credentials=False)


def router(config,root,require_token):
    routes=APIRouter(prefix="/api/v1/controller/workbench");holder={};lock=threading.Lock()
    def service(token):
        require_token(token)
        with lock:
            if "service" not in holder:holder["service"]=Workbench(root,config)
        return holder["service"]
    @routes.get("/jobs")
    def jobs(token:str|None=Header(default=None,alias="X-Controller-Token")):
        return dict(jobs=service(token).listing(),parameters=list(PARAMETERS))
    @routes.post("/jobs")
    def submit(request:BatchRequest,token:str|None=Header(default=None,alias="X-Controller-Token")):
        try:return service(token).submit(request)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    @routes.post("/jobs/{job_id}/{action}")
    def control(job_id:str,action:str,token:str|None=Header(default=None,alias="X-Controller-Token")):
        try:return service(token).control(job_id,action)
        except KeyError as exc:raise HTTPException(404,str(exc)) from exc
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc
    @routes.get("/jobs/{job_id}/export")
    def export(job_id:str,token:str|None=Header(default=None,alias="X-Controller-Token")):
        try:return service(token).export(job_id)
        except KeyError as exc:raise HTTPException(404,str(exc)) from exc
    return routes
