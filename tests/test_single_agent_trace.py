import logging

from game_theory_agent.agents.single.models import DecisionTrace, PersonaProfile, PromptTemplate
from game_theory_agent.agents.single.trace import JsonlTraceStore, open_sqlite_checkpointer


def trace(
    round_number: int,
    *,
    episode_id: str = "episode-1",
    company_id: str = "company_A",
    trace_version: str = "single-agent-trace-v1.2.0",
) -> DecisionTrace:
    return DecisionTrace(
        trace_version=trace_version,
        episode_id=episode_id,
        company_id=company_id,
        round=round_number,
        state_version=round_number - 1,
        status="terminal",
        model_id="nvidia/nemotron-3-super-120b-a12b:free",
        persona=PersonaProfile(),
    )


def test_jsonl_trace_store_round_trips_append_only_records(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces")

    store.append(trace(1))
    store.append(trace(2))

    loaded = store.read_episode("episode-1")
    assert [item.round for item in loaded] == [1, 2]
    assert (tmp_path / "traces" / "episode-1.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_jsonl_trace_store_reads_bounded_company_history_before_round(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces")
    store.append(trace(1, trace_version="single-agent-trace-v1.0.0"))
    store.append(trace(2, company_id="company_B"))
    store.append(trace(2))
    store.append(trace(3))
    store.append(trace(4))
    store.append(trace(5))
    store.append(trace(6))
    store.append(trace(7))
    store.append(trace(3, episode_id="episode-2"))

    loaded = store.read_company_before_round(
        "episode-1",
        "company_A",
        round_number=7,
        limit=3,
    )

    assert [item.round for item in loaded] == [4, 5, 6]
    assert {item.company_id for item in loaded} == {"company_A"}
    assert {item.episode_id for item in loaded} == {"episode-1"}
    assert loaded[0].trace_version == "single-agent-trace-v1.2.0"


def test_sqlite_checkpointer_is_created_under_the_requested_output_path(tmp_path):
    path = tmp_path / "checkpoints" / "agent.sqlite"

    saver = open_sqlite_checkpointer(path)
    try:
        assert path.is_file()
    finally:
        saver.conn.close()


def test_sqlite_checkpointer_explicitly_allows_local_agent_models(tmp_path, caplog):
    saver = open_sqlite_checkpointer(tmp_path / "agent.sqlite")
    try:
        payload = saver.serde.dumps_typed(PromptTemplate())
        with caplog.at_level(logging.WARNING, logger="langgraph.checkpoint.serde.jsonplus"):
            restored = saver.serde.loads_typed(payload)
        assert restored == PromptTemplate()
        assert "unregistered type" not in caplog.text
    finally:
        saver.conn.close()
