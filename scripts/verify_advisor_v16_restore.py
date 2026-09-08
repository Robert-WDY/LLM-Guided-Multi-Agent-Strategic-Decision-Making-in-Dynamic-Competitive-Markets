"""Compare persisted evidence and live read APIs before and after local restore."""
import sys,json,hashlib,sqlite3
from urllib.request import urlopen
from pathlib import Path
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'.local-state-v14-release';DEST=ROOT/'runs/advisor-v16'
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def get(path):
 with urlopen('http://127.0.0.1:8010/api/v1/controller/'+path,timeout=30) as response:return json.load(response)
def snapshot():
 jobs=get('workbench/jobs');assert all(j['status']=='complete' for j in jobs['jobs']);records={};offset=0
 while True:
  page=get(f'theory-lab/experiments?offset={offset}&limit=30');assert not page['issues']
  for row in page['experiments']:records[row['id']]=digest(get('theory-lab/experiments/'+row['id']))
  if page['next_offset'] is None:break
  offset=page['next_offset']
 exports={j['id']:digest(get('workbench/jobs/'+j['id']+'/export')) for j in jobs['jobs']}
 with sqlite3.connect(DATA/'sessions.sqlite3') as db:sessions=[list(r) for r in db.execute('SELECT document,sha256 FROM checkpoints ORDER BY sha256').fetchall()]
 files={p.relative_to(DATA).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'workbench').rglob('*.json')}
 return dict(concepts=records,jobs=digest(jobs),exports=exports,sessions=sessions,files=files,budget=status())
result=snapshot();path=DEST/'restore-before.json'
if sys.argv[1]=='before':
 assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(concepts=len(result['concepts']),jobs=len(result['exports']),files=len(result['files']),sessions=len(result['sessions']))))
else:
 assert result==json.loads(path.read_text(encoding='utf-8'))
 report=dict(passed=True,concepts=len(result['concepts']),completed_jobs=len(result['exports']),workbench_files=len(result['files']),ordinary_episodes=len(result['sessions']),all_api_and_hashes_equal=True,budget_restored=False,budget_reserved_cny=result['budget']['reserved_cny'])
 (DEST/'backup-restore.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
