import math

from game_theory_agent.market.protocols import (
    SplitMix64,
    canonical_json,
    derive_sub_seed,
    seed_material,
    sha256_hash,
)


def test_rng_protocol_fixed_vector():
    material = seed_material("rng-splitmix64-v1.0.0", 42, 1, "demand_noise", "", 0)
    sub_seed = derive_sub_seed("rng-splitmix64-v1.0.0", 42, 1, "demand_noise", "", 0)
    rng = SplitMix64(sub_seed)

    assert material.hex() == (
        "0015726e672d73706c69746d697836342d76312e302e30"
        "000000000000002a00000001000c64656d616e645f6e6f697365000000000000"
    )
    assert sub_seed == 5_451_024_094_550_306_501
    assert [rng.next_u64() for _ in range(3)] == [
        161_151_717_074_112_168,
        13_603_652_428_006_878_901,
        13_991_462_904_532_020_131,
    ]
    assert math.isclose(
        SplitMix64(sub_seed).normal_approx(), -0.8620774209361457, abs_tol=1e-15
    )


def test_canonical_json_and_hash_fixed_vector():
    payload = {"b": 2, "a": {"x": 1_000_000, "z": "fresh"}}
    assert canonical_json(payload) == '{"a":{"x":1000000,"z":"fresh"},"b":2}'
    assert sha256_hash(payload) == (
        "sha256:df0724c6e332985b425e07944be430d95c212085a9adfbc808c50caf17904aba"
    )
