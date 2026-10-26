# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MP Least Frequently Used eviction policy."""

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
    LFUEvictionPolicy,
)
from lmcache.v1.distributed.l2_adapters.config import (
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)


def _key(chunk_id: int) -> ObjectKey:
    """Create a test key with the requested chunk identifier."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="model",
        kv_rank=0,
    )


class TestLFUEvictionPolicy:
    """Verify frequency ordering and MP lifecycle behavior."""

    def test_repeatedly_touched_key_survives_cold_keys(self) -> None:
        policy = LFUEvictionPolicy()
        hot, cold1, cold2 = _key(1), _key(2), _key(3)
        policy.on_keys_created([hot, cold1, cold2])
        policy.on_keys_touched([hot, hot])

        actions = policy.get_eviction_actions(0.67)

        assert actions[0].keys == [cold2, cold1]

    def test_equal_frequency_uses_least_recent_order(self) -> None:
        policy = LFUEvictionPolicy()
        key1, key2, key3 = _key(1), _key(2), _key(3)
        policy.on_keys_created([key1, key2, key3])
        policy.on_keys_touched([key1, key2, key3])

        actions = policy.get_eviction_actions(1.0)

        assert actions[0].keys == [key3, key2, key1]

    def test_eligible_filter_skips_locked_key(self) -> None:
        policy = LFUEvictionPolicy()
        key1, key2 = _key(1), _key(2)
        policy.on_keys_created([key1, key2])

        actions = policy.get_eviction_actions(
            0.5, key_eligible_filter=lambda key: key != key2
        )

        assert actions[0].keys == [key1]

    def test_l1_write_lifecycle_counts_one_admission(self) -> None:
        policy = LFUEvictionPolicy()
        listener = L1EvictionPolicy(policy)
        admitted, cold = _key(1), _key(2)
        listener.on_l1_keys_reserved_write([admitted])
        listener.on_l1_keys_write_finished([admitted])
        policy.on_keys_created([cold])

        actions = policy.get_eviction_actions(0.5)

        assert actions[0].keys == [admitted]

    def test_removed_key_is_no_longer_tracked(self) -> None:
        policy = LFUEvictionPolicy()
        key = _key(1)
        policy.on_keys_created([key])

        policy.on_keys_removed([key])

        assert policy.get_num_tracked_keys() == 0
        assert policy.get_eviction_actions(1.0) == []


class TestLFUConfiguration:
    """Verify LFU selection through supported configuration surfaces."""

    def test_factory_creates_lfu_policy(self) -> None:
        policy = CreateEvictionPolicy(EvictionConfig(eviction_policy="LFU"))
        assert isinstance(policy, LFUEvictionPolicy)

    def test_l1_cli_accepts_lfu(self) -> None:
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        args = parser.parse_args(["--l1-size-gb", "1", "--eviction-policy", "LFU"])

        config = parse_args_to_config(args)

        assert config.eviction_config.eviction_policy == "LFU"

    def test_l2_adapter_config_accepts_lfu(self) -> None:
        parser = argparse.ArgumentParser()
        add_l2_adapters_args(parser)
        spec = {
            "type": "mock",
            "max_size_gb": 1,
            "mock_bandwidth_gb": 10,
            "eviction": {"eviction_policy": "LFU"},
        }
        args = parser.parse_args(["--l2-adapter", json.dumps(spec)])

        config = parse_args_to_l2_adapters_config(args)

        assert config.adapters[0].eviction_config is not None
        assert config.adapters[0].eviction_config.eviction_policy == "LFU"
