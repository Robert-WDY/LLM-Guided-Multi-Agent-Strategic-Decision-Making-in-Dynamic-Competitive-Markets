from copy import deepcopy

import pytest

from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import ConfigError


def test_v4_config_is_complete_and_stable(config):
    assert config.config_id == "market-v4-default"
    assert config.rounds == 10
    assert config.base_demand_orders == 12_000
    assert config.integer("company_initial", "cash_balance_cents") == 30_000_000
    assert config.config_sha256.startswith("sha256:")
    assert (
        MarketConfig.from_mapping(config.to_dict()).config_sha256
        == config.config_sha256
    )


def test_config_is_deeply_read_only(config):
    with pytest.raises(TypeError):
        config.data["market"]["rounds"] = 3


def test_config_rejects_bad_segment_weights(config):
    invalid = deepcopy(config.to_dict())
    invalid["consumer_choice"]["segments"]["price_sensitive"]["weight_ppm"] += 1
    with pytest.raises(ConfigError, match="sum to 1000000"):
        MarketConfig.from_mapping(invalid)
