# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the MP SIEVE eviction policy."""

# Standard
import argparse
import json

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import (
    CreateEvictionPolicy,
    SieveEvictionPolicy,
)
from lmcache.v1.distributed.internal_api import EvictionDestination
from lmcache.v1.distributed.l2_adapters.config import (
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)


def _key(chunk_id: int) -> ObjectKey:
    """Create a stable object key for one test chunk.

    Args:
        chunk_id: Integer encoded into the key hash.

    Returns:
        An object key with a fixed model and KV rank.
    """
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="model",
        kv_rank=0,
    )


def _victims(policy: SieveEvictionPolicy, ratio: float) -> list[ObjectKey]:
    """Request victims and return their keys.

    Args:
        policy: Policy under test.
        ratio: Fraction of resident keys requested for eviction.

    Returns:
        Keys contained in the single returned eviction action.
    """
    actions = policy.get_eviction_actions(ratio)
    assert len(actions) == 1
    return actions[0].keys


class TestSieveSelection:
    def test_touched_key_gets_second_chance(self) -> None:
        policy = SieveEvictionPolicy()
        key1, key2, key3 = _key(1), _key(2), _key(3)
        for key in (key1, key2, key3):
            policy.on_keys_created([key])
        policy.on_keys_touched([key1])

        assert _victims(policy, 0.67) == [key2, key3]

    def test_scan_clears_visited_bit_for_later_cycle(self) -> None:
        policy = SieveEvictionPolicy()
        key1, key2 = _key(1), _key(2)
        policy.on_keys_created([key1, key2])
        policy.on_keys_touched([key1])

        first_victim = _victims(policy, 0.5)
        policy.on_keys_removed(first_victim)

        assert _victims(policy, 1.0) == [key1]

    def test_batch_admission_prefers_suffix_for_eviction(self) -> None:
        policy = SieveEvictionPolicy()
        key1, key2, key3 = _key(1), _key(2), _key(3)

        policy.on_keys_created([key1, key2, key3])

        assert _victims(policy, 0.34) == [key3]

    def test_filter_skips_locked_key(self) -> None:
        policy = SieveEvictionPolicy()
        key1, key2, key3 = _key(1), _key(2), _key(3)
        for key in (key1, key2, key3):
            policy.on_keys_created([key])

        actions = policy.get_eviction_actions(
            0.34, key_eligible_filter=lambda key: key != key1
        )

        assert actions[0].keys == [key2]

    def test_ratio_and_destination_match_mp_contract(self) -> None:
        policy = SieveEvictionPolicy(default_destination=EvictionDestination.L2_CACHE)
        policy.on_keys_created([_key(1)])

        assert policy.get_eviction_actions(0.0) == []
        assert (
            policy.get_eviction_actions(0.01)[0].destination
            == EvictionDestination.L2_CACHE
        )

    def test_reservation_completion_does_not_refresh_queue_position(self) -> None:
        policy = SieveEvictionPolicy()
        listener = L1EvictionPolicy(policy)
        key1, key2 = _key(1), _key(2)

        listener.on_l1_keys_reserved_write([key1])
        listener.on_l1_keys_reserved_write([key2])
        listener.on_l1_keys_write_finished([key1])

        assert _victims(policy, 0.5) == [key1]


class TestSieveConfiguration:
    def test_factory_creates_sieve_policy(self) -> None:
        policy = CreateEvictionPolicy(EvictionConfig(eviction_policy="SIEVE"))

        assert isinstance(policy, SieveEvictionPolicy)

    def test_l1_cli_accepts_sieve(self) -> None:
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        args = parser.parse_args(["--l1-size-gb", "1", "--eviction-policy", "SIEVE"])

        config = parse_args_to_config(args)

        assert config.eviction_config.eviction_policy == "SIEVE"

    def test_l2_adapter_config_accepts_sieve(self) -> None:
        parser = argparse.ArgumentParser()
        add_l2_adapters_args(parser)
        spec = {
            "type": "mock",
            "max_size_gb": 1,
            "mock_bandwidth_gb": 10,
            "eviction": {"eviction_policy": "SIEVE"},
        }
        args = parser.parse_args(["--l2-adapter", json.dumps(spec)])

        config = parse_args_to_l2_adapters_config(args)

        assert config.adapters[0].eviction_config is not None
        assert config.adapters[0].eviction_config.eviction_policy == "SIEVE"
