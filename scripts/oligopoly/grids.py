"""Cell grids for the v21 oligopoly study (see experiment-specs/oligopoly-v1/PLAN.md)."""

from __future__ import annotations

# alias -> (provider, model id, thinking)
MODELS = {
    "db-mini": ("doubao", "doubao-seed-2-0-mini-260428", False),
    "db-lite": ("doubao", "doubao-seed-2-1-lite-260915", False),
    "db-pro": ("doubao", "doubao-seed-2-1-pro-260915", False),
    "ds-flash": ("deepseek", "deepseek-flash", False),
    "ds-pro": ("deepseek", "deepseek-v4-pro", False),
    "ds-flash-think": ("deepseek", "deepseek-flash", True),
}


def cell(name, firms, *, personas=None, communication=False, advisor=None,
         opening=None, advisor_firms=None, prompt_variant="default", rounds=100, window=30,
         seeds=range(3)):
    n = len(firms)
    return {
        "cell": name,
        "n": n,
        "firms": list(firms),
        "personas": list(personas or ["neutral"] * n),
        "communication": communication,
        "advisor": advisor,
        "opening": opening,
        "advisor_firms": advisor_firms,
        "prompt_variant": prompt_variant,
        "rounds": rounds,
        "window": window,
        "seeds": list(seeds),
    }


MAIN = dict(rounds=50, window=15)
S20 = range(100, 120)
S12 = range(200, 212)

E1 = [cell(f"{m}-base", [m] * 2, seeds=S20, **MAIN)
      for m in ("db-mini", "db-lite", "db-pro", "ds-flash", "ds-pro")]

E2 = []
for m in ("ds-flash", "db-lite"):
    E2 += [
        cell(f"{m}-comm", [m] * 2, communication=True, seeds=S20, **MAIN),
        cell(f"{m}-advisor-br", [m] * 2, advisor="br", seeds=S20, **MAIN),
        cell(f"{m}-advisor-full", [m] * 2, advisor="full", seeds=S20, **MAIN),
        cell(f"{m}-aggressive", [m] * 2, personas=["aggressive"] * 2, seeds=S20, **MAIN),
        cell(f"{m}-cooperative", [m] * 2, personas=["cooperative"] * 2, seeds=S20, **MAIN),
    ]

E3 = []
for m in ("ds-flash", "db-lite"):
    E3 += [
        cell(f"{m}-n4-base", [m] * 4, seeds=S12, **MAIN),
        cell(f"{m}-n4-comm", [m] * 4, communication=True, seeds=S12, **MAIN),
    ]
E3 += [
    cell("ds-flash-vs-br", ["ds-flash", "rule:myopic_br"], seeds=S20, **MAIN),
    cell("ds-flash-vs-db-lite", ["ds-flash", "db-lite"], seeds=S20, **MAIN),
]

# E4 (follow-up, specified after E1-E3 mechanism tables; see PLAN.md addendum)
S16 = range(300, 316)
E4 = []
for m in ("ds-flash", "db-lite", "ds-pro"):
    E4 += [
        cell(f"{m}-open-nash", [m] * 2, opening="nash", seeds=S16, **MAIN),
        cell(f"{m}-open-nash-comm", [m] * 2, opening="nash", communication=True, seeds=S16, **MAIN),
    ]
E4 += [cell("ds-flash-think-base", ["ds-flash-think"] * 2, seeds=range(400, 412), **MAIN)]

# E5: unilateral game-theory advisor (firm 0 only); E6: long horizon from Nash
E5 = [cell(f"{m}-advisor-br-firm0", [m] * 2, advisor="br", advisor_firms=[0],
           seeds=range(500, 520), **MAIN) for m in ("ds-flash", "db-lite")]
E6 = [cell("ds-flash-open-nash-long", ["ds-flash"] * 2, opening="nash",
           rounds=150, window=30, seeds=range(600, 610))]

# E7 (PLAN addendum C): long horizon on two more models; prompt-wording robustness
E7 = [cell(f"{m}-open-nash-long", [m] * 2, opening="nash", rounds=150, window=30,
           seeds=range(600, 610)) for m in ("db-lite", "ds-pro")]
E7 += [cell(f"{m}-base-alt", [m] * 2, prompt_variant="alt", seeds=range(700, 720), **MAIN)
       for m in ("ds-flash", "db-lite")]

PHASES: dict[str, list[dict]] = {
    "e7": E7,
    "e5": E5,
    "e6": E6,
    "e4": E4,
    "e1": E1,
    "e2": E2,
    "e3": E3,
    "pilot": [
        cell("ds-flash-base", ["ds-flash"] * 2, seeds=range(3)),
        cell("db-lite-base", ["db-lite"] * 2, seeds=range(3)),
        cell("ds-pro-base", ["ds-pro"] * 2, rounds=30, window=10, seeds=range(1)),
        cell("db-mini-base", ["db-mini"] * 2, rounds=30, window=10, seeds=range(1)),
    ],
}
