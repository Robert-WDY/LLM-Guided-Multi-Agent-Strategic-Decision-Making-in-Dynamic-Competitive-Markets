"""Fresh, fixed-size confirmation of the SmolLM numeric EconEvals result."""
from concurrent.futures import ThreadPoolExecutor

from markets import OUT,RAW,read,save
from numeric import CACHE,trial
from precision import stats
from results import summarize

SEEDS=range(155000,155256)


def analyze(rows):
    assert [r["seed"] for r in rows]==list(SEEDS)
    assert all(r["model"]=="smol" for r in rows)
    return {"discounted":summarize(rows,"econ",.99),"undiscounted":summarize(rows,"econ",1.),
            "precision":stats("econ",rows)}


if __name__=="__main__":
    jobs=[("econ","smol",s) for s in SEEDS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(trial,jobs))
    rows=[read(RAW/f"numeric-econ-smol-{s}.json.gz") for s in SEEDS]
    keys={e[1] for r in rows for calls in (r["prefix_calls"],*r["branch_calls"]) for e in calls}
    save(OUT/"replication-requests.json.gz",{k:read(CACHE/(k+".json")) for k in sorted(keys)})
    save(OUT/"replication.json.gz",rows)
    save(OUT/"replication-results.json",analyze(rows))
