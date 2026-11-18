"""Unweighted total profit to terminal on the same locked decisions."""
from report import OUT, ARMS, read, save, ci
import numpy as np


def results():
    out={}
    for phase in ("test","real"):
        for market in ("econ","native"):
            rows=read(OUT/f"{phase}-{market}-paths.json.gz")
            vals=[]
            for r in rows:
                p={name:sum(x[3] for x in r["paths"][name]["settled"]) for name in ARMS}
                g={m:(p[v["arm"]]-p["keep"])*v["use"]/(100 if market=="native" else 1)
                   for m,v in r["decision"].items()}
                vals.append((g["conditional"],g["conditional"]-g["generic"]))
            out[f"{phase}-{market}"]=dict(zip(("conditional_keep","conditional_generic"),map(ci,zip(*vals))))
    for label in ("empirical","empirical-replication"):
        vs_keep=[];vs_up=[]
        for r in read(OUT/f"{label}-paths.json.gz"):
            p={k:sum(v) for k,v in r["paths"].items()}
            v=p["finalist"] if r["gate"]["empirical"]!="agent_draft" else p["keep"]
            vs_keep.append(v-p["keep"])
            vs_up.append(v-p["up"])
        out[label]=dict(empirical_keep=ci(vs_keep),empirical_fixed_up=ci(vs_up))
    return out


if __name__=="__main__":
    save(OUT/"undiscounted.json",results())
