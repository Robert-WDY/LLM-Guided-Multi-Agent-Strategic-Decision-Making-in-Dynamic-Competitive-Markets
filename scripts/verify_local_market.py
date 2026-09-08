"""Cold HTTP acceptance of the local UI's actual API workflow; no provider calls."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get('MARKET_SMOKE_OUTPUT', 'runs/local-market-stage18-r2')

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = dict(os.environ, MARKET_CONFIG_PATH=str(ROOT / 'configs/market_v13_1_local.yaml'), MARKET_SESSION_DB=str(OUT / 'sessions.sqlite3'), MARKET_API_HOST='127.0.0.1', MARKET_API_PORT=str(port), MARKET_AGENT_GATEWAY_ENABLED='0', MARKET_PERSISTENCE='1', MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER='1', PYTHONPATH=os.pathsep.join((str(ROOT/'src'), str(ROOT/'.venv/Lib/site-packages'))))
    process = None
    logfile = (OUT/'server.log').open('w', encoding='utf-8')
    def start():
        nonlocal process
        process = subprocess.Popen([sys._base_executable, '-m', 'game_theory_agent.api'], cwd=ROOT, env=env, stdout=logfile, stderr=logfile, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for _ in range(150):
            if process.poll() is not None: raise RuntimeError('server exited; see server.log')
            try:
                if httpx.get(base+'/api/health').status_code == 200: return
            except httpx.TransportError: pass
            time.sleep(.1)
        raise RuntimeError('server did not start')
    def stop():
        if process and process.poll() is None:
            process.terminate(); process.wait(timeout=10)
    def req(method, path, **kwargs):
        response = httpx.request(method, base+path, timeout=180, **kwargs)
        if not response.is_success: raise RuntimeError(f'{path}: {response.status_code} {response.text[:2000]}')
        return response.json()
    def run(payload):
        state = payload['state']
        body = dict(run_id=f"ui:{state['round']}", expected_round=state['round'], expected_state_version=state['state_version'], expected_state_hash=state['state_hash'], max_rounds=1, authorize_real_model=False)
        result = req('POST', f"/api/v1/controller/episodes/{state['episode_id']}/coordinator-run", json=body)
        assert result['execution']['real_model']['required_calls'] == 0
        assert result['execution']['settled_round_count'] == 1
        return result, body
    results = []
    try:
        start()
        for advice in (False, True):
            episode = 'local-ui-advice' if advice else 'local-ui-rules'
            path = f'/api/v1/controller/episodes/{episode}'
            created = req('POST', '/api/episodes', json=dict(episode_id=episode, episode_seed=210091, max_rounds=5, information_mode='public', communication_mode='public_private', cooperation_mode='combined_v1', belief_mode='public_action_v1' if advice else 'off', opponent_model_mode='public_strategy_v1' if advice else 'off', utility_inference_mode='strategy_utility_v1' if advice else 'off', advisor_mode='strategic_market_v9' if advice else 'off'))
            assert created['state']['government'] is not None
            payload, body = run(created)
            payload, body = run(payload)
            ui = {'entryMode':'research', 'config':{'rounds':5,'seed':210091,'gameTheory':advice,'controllerToken':'must-not-export'},'agents':[]}
            req('POST', path+'/save', json={'ui_state':ui})
            evidence = req('GET', path+'/export')
            assert 'controllerToken' not in evidence['ui_state']['config']
            assert len(evidence['transitions']) == 2
            stop(); start()
            restored = req('POST', path+'/restore', json={})
            assert restored['state'] == payload['state']
            assert restored['checkpoint']['ui_state']['config']['gameTheory'] == advice
            repeated = req('POST', path+'/coordinator-run', json=body)
            assert repeated == payload
            payload = restored
            while not payload['state']['terminal']: payload, body = run(payload)
            final = req('GET', path+'/export')
            assert len(final['transitions']) == 5
            assert len(final['coordinator_runs']) == 5
            (OUT/f'{episode}.json').write_text(json.dumps(final,ensure_ascii=False),encoding='utf-8')
            results.append({'episode':episode,'rounds':5,'advice':advice,'cold_restart':True,'repeat_idempotent':True,'terminal_hash':payload['state']['state_hash']})
        long = req('POST','/api/episodes',json={'episode_id':'local-long','max_rounds':60,'episode_seed':210092})
        assert long['state']['max_rounds'] == 60
        long_started = time.perf_counter()
        for _ in range(60): long, _ = run(long)
        assert long['state']['terminal']
        long_seconds = round(time.perf_counter()-long_started, 3)
        stop(); start()
        long_restored = req('POST','/api/v1/controller/episodes/local-long/restore',json={})
        assert long_restored['state'] == long['state']
        listed = req('GET','/api/v1/controller/saved-episodes')
        assert len(listed['episodes']) == 3
        report={'passed':True,'model_calls':0,'episodes':results,'long_creation_accepted':True,'long_http_rounds':60,'long_http_seconds':long_seconds,'long_terminal_restore':True,'database_bytes':(OUT/'sessions.sqlite3').stat().st_size,'browser_qa':False}
        (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:
        stop(); logfile.close()

if __name__ == '__main__': main()
