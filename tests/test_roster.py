import pytest

from game_theory_agent.orchestration.roster import (
    COMPANY_IDS,
    DEFAULT_COMPANIES,
    parse_company_list,
    require_subset,
    validate_company_roster,
)


def test_default_roster_is_four_companies_and_ten_are_named():
    assert DEFAULT_COMPANIES == (
        "company_A",
        "company_B",
        "company_C",
        "company_D",
    )
    assert COMPANY_IDS[-1] == "company_J"
    assert len(COMPANY_IDS) == 10


def test_validate_company_roster_accepts_two_to_ten():
    eight = validate_company_roster(COMPANY_IDS[:8])
    assert eight == COMPANY_IDS[:8]
    assert validate_company_roster(COMPANY_IDS) == COMPANY_IDS


def test_validate_company_roster_rejects_unknown_and_duplicates():
    with pytest.raises(ValueError, match="unknown companies"):
        validate_company_roster(("company_A", "company_Z"))
    with pytest.raises(ValueError, match="duplicates"):
        validate_company_roster(("company_A", "company_B", "company_A"))
    with pytest.raises(ValueError, match="2 to 10"):
        validate_company_roster(("company_A",))


def test_agent_companies_must_be_subset_of_roster():
    roster = validate_company_roster(parse_company_list("company_A,company_B,company_C"))
    assert require_subset(("company_A", "company_C"), roster, flag="--agent-companies")
    with pytest.raises(ValueError, match="--agent-companies"):
        require_subset(("company_D",), roster, flag="--agent-companies")


def test_run_agents_cli_accepts_six_company_roster():
    from game_theory_agent.run_agents import _parser

    args = _parser().parse_args(
        [
            "--provider",
            "mock",
            "--companies",
            "company_A,company_B,company_C,company_D,company_E,company_F",
            "--agent-companies",
            "company_A,company_E",
        ]
    )
    roster = validate_company_roster(parse_company_list(args.companies))
    selected = require_subset(
        parse_company_list(args.agent_companies),
        roster,
        flag="--agent-companies",
    )
    assert roster[-1] == "company_F"
    assert selected == ("company_A", "company_E")
