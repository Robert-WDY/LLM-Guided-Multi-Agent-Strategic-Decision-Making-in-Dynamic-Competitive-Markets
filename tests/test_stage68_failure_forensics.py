from game_theory_agent.experiments.stage68_failure_forensics import (
    _shapley_decomposition,
)


def test_stage68_shapley_helper_is_importable_for_replay_forensics():
    # The full deterministic decomposition is run against recorded Episodes by
    # the experiment entrypoint.  This test keeps the public helper contract
    # visible without duplicating a multi-minute audit in the unit suite.
    assert callable(_shapley_decomposition)
