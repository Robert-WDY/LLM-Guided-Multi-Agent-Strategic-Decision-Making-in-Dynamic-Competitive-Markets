"""Verify all primary effects with NumPy only; optionally replay independent market transitions."""
import argparse
from collections import Counter
import numpy as np

from publication_stats import OUT, read, paired, real_sample


def verify_tables():
    result=read(OUT/"results.json")
    delayed=read(OUT/"delayed-results.json")
    for m in ("econ","native"):
        for discount in (.99,1.):
            assert paired(m,discount)==result[m][str(discount)]
        rows=read(OUT/f"delayed-test-{m}.json.gz")
        differences=[]
        for r in rows:
            effects=[]
            for c in r["cases"]:
                t=c["public"]["round"]
                assert len(c["public"]["prices"])==t-1
                p=np.array(c["profit"])
                assert p.shape==(3,61-t)
                a,b=(c["selected"][k] for k in ("inferred","uniform"))
                effects.append((p[a]-p[b])@(.99**np.arange(61-t))/(100 if m=="native" else 1))
            differences.append(np.mean(effects))
        diff=np.array(differences)
        idx=np.random.default_rng(4129).integers(len(diff),size=(12000,len(diff)))
        row=delayed[m]["test"]["inferred-uniform"]
        np.testing.assert_allclose(np.quantile(diff[idx].mean(1),[.0125,.9875]),row["ci975"])
        np.testing.assert_allclose(diff.mean(),row["mean"])
    assert real_sample("native",False)==result["fresh"] and real_sample("econ",True)==result["archive"]
    fresh=read(OUT/"fresh-native.json.gz")
    assert Counter(r["provider"] for r in fresh)=={"deepseek":4,"doubao":4}
    assert [r["seed"] for r in fresh]==list(range(105000,105008))
    budget=read(OUT/"budget.json")
    assert budget["initial_micro_cny"]+sum(row[-1] for row in budget["rows"])==budget["final_micro_cny"]<=50_000_000
    print("Primary outcomes, seed-cluster intervals, real transfer, model allocation and budget verified.")


def replay():
    import paired as worlds
    import delayed
    from public_value_selector import verify as verify_prior
    from signal_policy import verify as verify_signal
    import unseen
    from public_price_value.audit import replay as replay_real
    verify_prior()
    verify_signal()
    assert delayed.snapshot()==read(OUT/"delayed-freeze.json")
    for m in ("econ","native"):
        for phase in ("train","cal","test"):
            rows=read(OUT/f"{phase}-{m}.json.gz")
            for r in (rows[0],rows[-1]):
                assert r==worlds.episode((phase,m,r["seed"]))
        for phase in ("train","test"):
            rows=read(OUT/f"delayed-{phase}-{m}.json.gz")
            for r in (rows[0],rows[-1]):
                assert r==delayed.trial((phase,m,r["seed"]))
    unseen_rows=read(OUT/"unseen-paths.json.gz")
    for r in (unseen_rows[0],unseen_rows[1],unseen_rows[-2],unseen_rows[-1]):
        assert r==unseen.trial((r["seed"],r["learns"]))
    assert unseen.results(unseen_rows)==read(OUT/"unseen.json")
    assert all(replay_real(r) for r in read(OUT/"fresh-native.json.gz"))
    print("Frozen models, sampled branch regeneration and all eight new real triple paths verified.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    verify_tables()
    if args.replay:
        replay()
