from game_theory_agent.market import MarketEnv
from game_theory_agent.market.replay import (
    EpisodeManifest,
    JsonlTransitionLogger,
    MarketTransition,
    replay,
    verify_replay,
)


def test_manifest_jsonl_and_hash_verified_replay(
    config, env, initial_state, make_actions, tmp_path
):
    manifest = EpisodeManifest.create(env, initial_state, experiment_id="exp-test")
    transitions = []
    joint_actions = []
    state = initial_state
    for index in range(3):
        actions = make_actions(state, nonce=str(index))
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        transitions.append(MarketTransition.create(state, actions, result))
        joint_actions.append(actions)
        state = result.state_after

    logger = JsonlTransitionLogger(tmp_path / "events.jsonl")
    for transition in transitions:
        logger.append(transition)
    loaded = logger.read_all()

    assert [item.state_after.state_hash for item in loaded] == [
        item.state_after.state_hash for item in transitions
    ]
    assert (
        replay(MarketEnv(config), initial_state, joint_actions)[-1].state_hash
        == state.state_hash
    )
    assert (
        verify_replay(MarketEnv(config), manifest, loaded)[-1].state_hash
        == state.state_hash
    )


def test_state_serialization_preserves_company_order(initial_state):
    restored = type(initial_state).from_dict(initial_state.to_dict())
    assert restored == initial_state
    assert restored.company_ids == initial_state.company_ids
