"""Evaluate the actual ReliableRequest research path and its rejected finalist to terminal."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from trials import OUT, RAW, read, save
from reliable_v19_common import source, MarketEnv, actions, step, ReliableRequest, advise
from game_theory_agent.market import CompanyAction


def terminal(config, state, memory, seed, initial):
    remaining = state.rounds_remaining
    env = MarketEnv(config)
    env.load_state(state)
    mem = deepcopy(memory)
    profits, hashes = [], []
    while not state.terminal:
        aa, mem = actions(config, state, mem, seed)
        if not profits:
            aa["company_A"] = CompanyAction.from_dict(initial)
        state = step(env, state, aa)
        profits.append(state.company("company_A").financial.round_profit_cents / 100)
        hashes.append(state.state_hash)
    assert len(profits) == remaining
    return {"profit": profits, "hashes": hashes}


def one(seed):
    path = RAW / f"native-advisor-{seed}.json.gz"
    if path.exists():
        return
    regime = ("normal", "recession", "tight_supply")[seed % 3]
    c, s, mem, draft, history = source(regime, 2, seed)
    public = [{k: row["state"][k] for k in ("round", "prices")} for row in history]
    assert s.round == 5 and public[-1]["round"] == 4
    raw = draft.to_dict()
    proposals = {}
    candidates = {"agent_draft": raw}
    for mode in ("conditional", "generic"):
        result = advise(c, s, ReliableRequest(draft_action=raw, goal="profit", mode="research",
                        seed=17, discount=.99, response_model=mode, max_candidates=6,
                        step_budget=500, public_history=public))
        assert result["draft_action"] == raw and result["search"]["complete"]
        best = max((r for r in result["rankings"] if r["candidate"]["id"] != "agent_draft"),
                   key=lambda r: (r["safe"], r["score"]))["candidate"]
        assert result["recommended_id"] in ("agent_draft", best["id"])
        candidates[best["id"]] = best["action"]
        proposals[mode] = {"accepted": result["recommended_id"] != "agent_draft", "id": best["id"],
                           "steps": result["search"]["used_steps"],
                           "gate": [g["lower_bound"] for g in result["validation"]]}
    seal = RAW / f"sealed-native-advisor-{seed}.json"
    if seal.exists():
        assert read(seal) == proposals
    else:
        save(seal, proposals)
    paths = {name: terminal(c, s, mem, seed, action) for name, action in candidates.items()}
    assert paths["agent_draft"] == terminal(c, s, mem, seed, raw)
    save(path, dict(seed=seed, regime=regime, source_hash=s.state_hash, proposals=proposals, paths=paths))


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as ex:
        list(ex.map(one, range(95000, 95024)))
    save(OUT / "native-advisor-paths.json.gz", [read(RAW / f"native-advisor-{s}.json.gz") for s in range(95000, 95024)])
