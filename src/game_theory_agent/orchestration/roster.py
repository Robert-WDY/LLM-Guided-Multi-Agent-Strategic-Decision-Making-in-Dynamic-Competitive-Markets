"""Canonical 2–10 company roster used by CLI, API, and stability tests."""

from __future__ import annotations


COMPANY_IDS: tuple[str, ...] = tuple(
    f"company_{chr(ord('A') + index)}" for index in range(10)
)
DEFAULT_COMPANIES: tuple[str, ...] = COMPANY_IDS[:4]


def parse_company_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_company_roster(
    companies: tuple[str, ...] | list[str],
    *,
    min_count: int = 2,
    max_count: int = 10,
    allowed: tuple[str, ...] = COMPANY_IDS,
) -> tuple[str, ...]:
    roster = tuple(companies)
    if not min_count <= len(roster) <= max_count:
        raise ValueError(
            f"company roster must contain {min_count} to {max_count} companies"
        )
    if len(set(roster)) != len(roster):
        raise ValueError("company roster contains duplicates")
    unknown = [company_id for company_id in roster if company_id not in allowed]
    if unknown:
        raise ValueError(
            "unknown companies: "
            f"{unknown}; expected a subset of {list(allowed)}"
        )
    return roster


def require_subset(
    selected: tuple[str, ...],
    roster: tuple[str, ...],
    *,
    flag: str,
) -> tuple[str, ...]:
    extra = [company_id for company_id in selected if company_id not in roster]
    if not selected or extra:
        raise ValueError(
            f"{flag} must be a non-empty subset of the company roster; "
            f"unexpected {extra}"
        )
    return selected
