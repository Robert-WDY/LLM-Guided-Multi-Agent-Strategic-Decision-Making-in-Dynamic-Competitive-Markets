"""Pinned upstream market, file IO and explicit experiment boundaries."""
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("ECONEVALS_ROOT", ROOT / ".tmp/econ-evals-upstream"))
REVISION = "e1f2a40fec96f0d27f5414873c4310f2b5c51935"
if subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() != REVISION:
    raise ValueError("EconEvals revision mismatch")
if subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "HEAD", "--", "econ_evals"], text=True):
    raise ValueError("Upstream tracked source has changed")
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "src"))
from econ_evals.experiments.pricing import pricing_market_logic_multiproduct as market

OUT = ROOT / "experiment-results/econevals-pricing-v1"
RAW = ROOT / "runs/econevals-pricing-v1"
for folder in (OUT, RAW):
    folder.mkdir(exist_ok=True, parents=True)


def save(path, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None if path.suffix == ".gz" else 2).encode()
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    temp.replace(path)


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

