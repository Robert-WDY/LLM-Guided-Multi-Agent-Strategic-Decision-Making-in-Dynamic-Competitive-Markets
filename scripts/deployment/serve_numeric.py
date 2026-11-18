"""Run both locked models on loopback for the numeric study; own their entire process lifetime."""
from contextlib import ExitStack
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from urllib.request import ProxyHandler,build_opener

from numeric import SOURCES

ROOT=Path(__file__).resolve().parents[2]
MODELS=ROOT/".tmp/independent-models"


def run(stages):
    flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    opener=build_opener(ProxyHandler({}))
    exe=shutil.which("llama-server")
    if not exe:
        raise RuntimeError("llama-server unavailable")
    if os.name=="nt":
        # Keep our managed processes separate from other applications using the shared installation.
        src=Path(exe);folder=MODELS/"runtime";folder.mkdir(exist_ok=True)
        for f in [src,*src.parent.glob("*.dll")]:
            dest=folder/("pricing-worker.exe" if f==src else f.name)
            if not dest.exists():
                try:
                    os.link(f,dest)
                except OSError:
                    shutil.copy2(f,dest)
        exe=str(folder/"pricing-worker.exe")
    with ExitStack() as cleanup:
        for name,(_,_,_,port) in SOURCES.items():
            log=cleanup.enter_context((MODELS/f"{name}-confirm.log").open("ab"))
            proc=subprocess.Popen([exe,"-m",str(MODELS/(name+".gguf")),"--alias",name,"--host","127.0.0.1",
                "--port",str(port),"-c","2048","-np","2","-ngl","99","-t","2","--cache-ram","0"],
                creationflags=flags,stdout=log,stderr=subprocess.STDOUT)
            def stop_owned(p=proc):
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=30)
            cleanup.callback(stop_owned)
            for _ in range(90):
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} server failed to start; see its local log")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/health",timeout=1) as res:
                        if res.status==200:
                            break
                except OSError:
                    time.sleep(.5)
            else:
                raise RuntimeError(f"{name} model did not become ready")
        for script in stages:
            for attempt in range(3):
                done=subprocess.run([sys.executable,str(Path(__file__).with_name(script))],cwd=ROOT)
                if done.returncode==0:
                    break
                time.sleep(3)
            else:
                raise RuntimeError(f"{script} failed after bounded retries")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("stages",nargs="*",default=["numeric","adoption","numeric_stopping"])
    stages=parser.parse_args().stages
    if any(s not in ("numeric","adoption","numeric_stopping","replicate") for s in stages):
        parser.error("unknown pricing experiment")
    for attempt in range(4):
        try:
            run([s+".py" for s in stages])
            break
        except RuntimeError:
            if attempt==3:
                raise
            time.sleep(4)
