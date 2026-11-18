"""Run numeric study stages; only cache misses start a managed CPU model."""

from pathlib import Path
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / ".tmp/independent-models"


def run(stages, offline=False):
    env = dict(
        os.environ, PRICING_NUMERIC_BACKEND="offline" if offline else "managed-cpu"
    )
    for script in stages:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name(script))],
            cwd=ROOT,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages", nargs="*", default=["numeric", "adoption", "numeric_stopping"]
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require cached responses; never open a model service",
    )
    args = parser.parse_args()
    stages = args.stages
    if any(
        s not in ("numeric", "adoption", "numeric_stopping", "replicate")
        for s in stages
    ):
        parser.error("unknown pricing experiment")
    run([s + ".py" for s in stages], offline=args.offline)
