"""Durable, conservative cash reservations against the owner's authorized budget.

Reservations are never refunded automatically, including failed/unknown calls.
Session backups intentionally never roll this ledger back.
"""
from datetime import date
from contextlib import contextmanager, closing
from pathlib import Path
import json
import sqlite3
import uuid

TOTAL = 10_000_000
HISTORICAL = 7_008_000
HISTORICAL_ESTIMATE = 1_093_593
RESERVATION = 420_000  # 128k input * 3 + 4k output * 9 microunits
PRICE_VALID_THROUGH = '2026-09-08'
LEDGER = Path(__file__).resolve().parents[2] / '.local-state/model-budget.sqlite3'

class LocalBudgetError(RuntimeError): pass

def protected(config):
    return config.config_id == "market-v13-1-autonomous-local" or bool(config.data.get("local_budget_protected"))

def initialize(path=LEDGER):
    """Explicit first installation only; a marker prevents silent reset after loss."""
    path = Path(path)
    marker = path.with_suffix('.initialized')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): return status(path)
    if marker.exists(): raise LocalBudgetError('费用账本丢失；禁止自动重置额度。请保留备份并核对账单。')
    # Exclusive creation; concurrent installers must not create duplicate history.
    with path.open('xb'): pass
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE budget (id INTEGER PRIMARY KEY CHECK(id=1), total INTEGER NOT NULL, historical INTEGER NOT NULL)')
        db.execute('INSERT INTO budget VALUES (1, ?, ?)', (TOTAL, HISTORICAL))
        db.execute('CREATE TABLE calls (id TEXT PRIMARY KEY, reserved INTEGER NOT NULL, status TEXT NOT NULL, actual INTEGER, created TEXT DEFAULT CURRENT_TIMESTAMP)')
    marker.write_text('Budget initialized; never restore an older budget ledger.\n', encoding='utf-8')
    return status(path)

@contextmanager
def _connect(path):
    path = Path(path)
    if not path.is_file(): raise LocalBudgetError('真实模型费用账本尚未初始化，调用已阻止；规则模拟不受影响。')
    with closing(sqlite3.connect(path.as_uri()+'?mode=rw', uri=True, timeout=15)) as db, db:
        yield db

def _totals(db):
    row = db.execute('SELECT total, historical FROM budget WHERE id=1').fetchone()
    approved = TOTAL
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='budget_authorizations'").fetchone():
        approved = db.execute('SELECT coalesce(max(total),?) FROM budget_authorizations', (TOTAL,)).fetchone()[0]
    if row != (approved, HISTORICAL): raise LocalBudgetError('费用账本额度或历史保留额不匹配，已阻止调用。')
    if db.execute('SELECT count(*) FROM calls WHERE reserved <= 0 OR reserved > ?', (RESERVATION,)).fetchone()[0]: raise LocalBudgetError('费用账本损坏。')
    if db.execute('SELECT count(*) FROM calls WHERE actual > reserved').fetchone()[0]: raise LocalBudgetError('已有调用超出预留；必须核对账单后才能继续。')
    added, count = db.execute('SELECT coalesce(sum(reserved),0), count(*) FROM calls').fetchone()
    return added, count

def status(path=LEDGER):
    try:
        with _connect(path) as db:
            added, count = _totals(db)
            total = db.execute('SELECT total FROM budget WHERE id=1').fetchone()[0]
        remaining = max(0, total-HISTORICAL-added)
        valid = date.today().isoformat() <= PRICE_VALID_THROUGH
        return dict(ready=valid and remaining >= RESERVATION, total_cny=total/1e6, reserved_cny=(HISTORICAL+added)/1e6, remaining_cny=remaining/1e6, historical_estimate_cny=HISTORICAL_ESTIMATE/1e6, maximum_call_cny=RESERVATION/1e6, new_calls=count, model='deepseek-v4-flash', price_valid_through=PRICE_VALID_THROUGH, message='已保留失败及未知调用的额度。' if valid else '价格快照已过期；请重新核对价格后更新保护配置。')
    except (sqlite3.Error, OSError, LocalBudgetError) as exc:
        return dict(ready=False, message=str(exc), total_cny=10)

def reserve(path=LEDGER, amount=RESERVATION):
    if type(amount) is not int or not 0 < amount <= RESERVATION:
        raise LocalBudgetError("无效的调用费用上界。")
    if date.today().isoformat() > PRICE_VALID_THROUGH: raise LocalBudgetError('价格快照已过期，真实模型调用已阻止；规则模拟可继续。')
    try:
        with _connect(path) as db:
            db.execute('BEGIN IMMEDIATE')
            added, _ = _totals(db)
            total = db.execute('SELECT total FROM budget WHERE id=1').fetchone()[0]
            if HISTORICAL+added+amount > total: raise LocalBudgetError('已授权累计预算剩余额度不足；不会继续调用真实模型。')
            call_id = str(uuid.uuid4())
            db.execute('INSERT INTO calls (id,reserved,status) VALUES (?,?,?)', (call_id, amount, 'reserved_unknown'))
        return call_id
    except (sqlite3.Error, OSError) as exc: raise LocalBudgetError('无法持久化费用预留，未发起模型调用。') from exc

def authorize_increase(total_cny, authorization, path=LEDGER):
    """Explicit owner authorization only; preserve every old reservation and call."""
    if type(total_cny) is not int or not 10 < total_cny <= 30 or not isinstance(authorization,str) or not authorization.strip():
        raise LocalBudgetError('需要明确授权且累计上限不超过30元。')
    with _connect(path) as db:
        db.execute('BEGIN IMMEDIATE');_totals(db)
        old = db.execute('SELECT total FROM budget WHERE id=1').fetchone()[0]
        if total_cny*1_000_000 < old:raise LocalBudgetError('禁止通过授权操作回退额度。')
        db.execute('CREATE TABLE IF NOT EXISTS budget_authorizations (total INTEGER PRIMARY KEY, previous_total INTEGER NOT NULL, authorization TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP)')
        db.execute('INSERT OR IGNORE INTO budget_authorizations(total,previous_total,authorization) VALUES (?,?,?)',(total_cny*1_000_000,old,authorization))
        db.execute('UPDATE budget SET total=? WHERE id=1',(total_cny*1_000_000,))
    return status(path)


class GuardedCompletions:
    def __init__(self, delegate, path=LEDGER, *, compact=False, model_name="deepseek-v4-flash"):
        self.delegate, self.path, self.compact = delegate, path, compact
        if model_name not in {"deepseek-v4-flash","doubao-seed-2-0-lite-260215"}:raise LocalBudgetError("未核价模型禁止调用。")
        self.model_name=model_name
    async def create(self, **kwargs):
        # Only the narrow text/JSON, non-thinking request shape is price bounded.
        if kwargs.get('model') != self.model_name or kwargs.get('stream') is not False or kwargs.get('extra_body') != {'thinking':{'type':'disabled'}}: raise LocalBudgetError('本机费用保护仅支持已核价模型的非思考文本模式。')
        if set(kwargs)-{"model","messages","stream","max_tokens","temperature","top_p","response_format","extra_body"}:
            raise LocalBudgetError("预算协议不允许工具、多输出或其他未核价参数。")
        if kwargs.get("response_format",{"type":"json_object"})!={"type":"json_object"}:
            raise LocalBudgetError("只允许不附加额外schema的JSON文本。")
        messages = kwargs.get('messages', [])
        if not messages or any(not isinstance(m,dict) or set(m)!={"role","content"} or m.get("role") not in {"system","user","assistant"} or not isinstance(m.get('content'), str) for m in messages): raise LocalBudgetError('不支持的模型消息格式。')
        if len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) > 124_000: raise LocalBudgetError('提示内容超过本机费用保护长度上限，未发起调用。')
        if type(kwargs.get('max_tokens')) is not int or not 0 < kwargs['max_tokens'] <= 4000: raise LocalBudgetError('模型输出长度超过预算保护上限。')
        reservation = RESERVATION
        input_price,output_price=(3,9) if self.model_name=="deepseek-v4-flash" else (2,11)
        if self.compact:
            size=len(json.dumps(messages,ensure_ascii=False).encode("utf-8"))
            if size>8000 or len(messages)>8 or kwargs["max_tokens"]>1024:
                raise LocalBudgetError("紧凑决策超出8000字节/8消息/1024输出的预算边界。")
            # UTF-8 bytes bound byte-level tokens; 4096 extra tokens cover message framing.
            # Successful/failed reservations remain spent in the conservative ledger.
            input_price,output_price=(3,9) if self.model_name=="deepseek-v4-flash" else (1,4)
            reservation=(size+4096)*input_price+kwargs["max_tokens"]*output_price
        call_id = reserve(self.path, reservation)
        # No automatic transport/schema retry is configured by the hosted runtime.
        result = await self.delegate.create(**kwargs)
        usage = getattr(result, 'usage', None)
        prompt, completion = getattr(usage,'prompt_tokens',None), getattr(usage,'completion_tokens',None)
        if self.model_name!="deepseek-v4-flash" and type(prompt) is int and prompt>32000:input_price,output_price=2,11
        actual = prompt*input_price+completion*output_price if type(prompt) is int and type(completion) is int and prompt >= 0 and completion >= 0 else None
        if actual is not None and self.model_name!="deepseek-v4-flash":
            pi,po=(6,36) if prompt<=32000 else (9,54) if prompt<=128000 else (18,108)
            actual=(prompt*pi+completion*po+9)//10
        with _connect(self.path) as db:
            db.execute('UPDATE calls SET status=?, actual=? WHERE id=?', ('completed' if actual is not None else 'completed_usage_unknown', actual, call_id))
        if actual is not None and actual > reservation: raise LocalBudgetError('供应商报告用量超出预留上限，已保留全部额度；需要核对账单。')
        return result


class GuardedResponses:
    """Bound the existing Doubao Responses client with the same durable ledger."""
    def __init__(self,delegate,path=LEDGER):
        self.delegate,self.path=delegate,path

    async def create(self,**kwargs):
        from types import SimpleNamespace
        if set(kwargs)-{"model","input","max_output_tokens","extra_body","temperature","top_p"} or not isinstance(kwargs.get("input"),str):
            raise LocalBudgetError("豆包仅支持无工具、无缓存的纯文本响应。")
        original=self.delegate
        class Adapter:
            async def create(self,**_bounded):
                response=await original.create(**kwargs)
                usage=getattr(response,"usage",None)
                return SimpleNamespace(original=response,usage=SimpleNamespace(
                    prompt_tokens=getattr(usage,"input_tokens",None),completion_tokens=getattr(usage,"output_tokens",None)))
        result=await GuardedCompletions(Adapter(),self.path,model_name="doubao-seed-2-0-lite-260215").create(
            model=kwargs.get("model"),messages=[{"role":"user","content":kwargs["input"]}],stream=False,
            max_tokens=kwargs.get("max_output_tokens"),extra_body=kwargs.get("extra_body"))
        return result.original
