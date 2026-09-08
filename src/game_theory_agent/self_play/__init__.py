"""Final strategic-market self-play research API."""

from .contracts import FrozenPromotionCandidate, StrategyVersion
from .policies import (
    build_strategy_action,
    operational_strategy_ids,
    paired_partner,
    strategy_versions,
)
from .tournament import COMPANY_IDS, role_rotated_matchups, run_episode

__all__ = [
    "COMPANY_IDS",
    "FrozenPromotionCandidate",
    "StrategyVersion",
    "build_strategy_action",
    "operational_strategy_ids",
    "paired_partner",
    "role_rotated_matchups",
    "run_episode",
    "strategy_versions",
]
