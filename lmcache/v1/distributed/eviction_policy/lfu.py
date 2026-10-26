# SPDX-License-Identifier: Apache-2.0
"""Least Frequently Used eviction policy for MP mode."""

# Standard
from collections import OrderedDict
from collections.abc import Callable
import threading

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)


class LFUEvictionPolicy(EvictionPolicy):
    """Evict infrequently accessed keys, breaking ties by recency.

    Each key starts at frequency one. Explicit touches and repeated creations
    increase its frequency. Within one frequency, the least recently accessed
    key is selected first. Write reservation and completion notifications count
    as one admission so MP's two-phase write lifecycle does not inflate usage.

    Thread Safety:
        Every state transition is protected by one lock.
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ) -> None:
        """Initialize an empty LFU policy.

        Args:
            default_destination: Destination used when no destination has been
                registered by an eviction controller.
        """
        self._lock = threading.Lock()
        self._frequency_to_keys: dict[int, OrderedDict[ObjectKey, None]] = {}
        self._key_to_frequency: dict[ObjectKey, int] = {}
        self._pending_admissions: set[ObjectKey] = set()
        self._destinations: list[EvictionDestination] = []
        self._default_destination = default_destination

    def register_eviction_destination(self, destination: EvictionDestination) -> None:
        """Register an eviction destination.

        Args:
            destination: Destination to use for returned eviction actions.
        """
        with self._lock:
            if destination not in self._destinations:
                self._destinations.append(destination)

    def on_keys_reserved(self, keys: list[ObjectKey]) -> None:
        """Track reserved keys without double-counting completed writes.

        Args:
            keys: Keys reserved for creation.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._key_to_frequency:
                    self._increment_frequency(key)
                else:
                    self._add_key(key)
                self._pending_admissions.add(key)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Track newly resident keys.

        Args:
            keys: Keys that became resident after a completed write.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._pending_admissions:
                    self._pending_admissions.remove(key)
                elif key in self._key_to_frequency:
                    self._increment_frequency(key)
                else:
                    self._add_key(key)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Increase the frequency of resident keys that were accessed.

        Args:
            keys: Resident keys that were accessed.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                if key in self._key_to_frequency:
                    self._increment_frequency(key)

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Stop tracking removed keys.

        Args:
            keys: Keys confirmed deleted by the L1 manager or L2 adapter.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                self._remove_key(key)
                self._pending_admissions.discard(key)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Choose least-frequently used keys for an MP eviction cycle.

        Args:
            expected_ratio: Approximate fraction of resident keys to evict,
                clamped to ``[0.0, 1.0]``.
            key_eligible_filter: Optional predicate used to skip locked or
                otherwise ineligible keys.
            cache_salt: Ignored because LFU is a global, non-isolated policy.

        Returns:
            Zero or one eviction action containing eligible keys in ascending
            frequency and least-recent order. A positive ratio selects at least
            one key when possible.
        """
        del cache_salt
        with self._lock:
            tracked_count = len(self._key_to_frequency)
            if tracked_count == 0:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(tracked_count * expected_ratio)
            if expected_ratio > 0 and target_count == 0:
                target_count = 1
            if target_count == 0:
                return []

            victims: list[ObjectKey] = []
            for frequency in sorted(self._frequency_to_keys):
                for key in self._frequency_to_keys[frequency]:
                    if key_eligible_filter is not None and not key_eligible_filter(key):
                        continue
                    victims.append(key)
                    if len(victims) == target_count:
                        break
                if len(victims) == target_count:
                    break

            if not victims:
                return []

            destination = (
                self._destinations[0]
                if self._destinations
                else self._default_destination
            )
            return [EvictionAction(keys=victims, destination=destination)]

    def get_num_tracked_keys(self) -> int:
        """Return the number of resident keys tracked by the policy."""
        with self._lock:
            return len(self._key_to_frequency)

    def _add_key(self, key: ObjectKey) -> None:
        self._key_to_frequency[key] = 1
        self._frequency_to_keys.setdefault(1, OrderedDict())[key] = None

    def _increment_frequency(self, key: ObjectKey) -> None:
        frequency = self._key_to_frequency[key]
        keys = self._frequency_to_keys[frequency]
        del keys[key]
        if not keys:
            del self._frequency_to_keys[frequency]

        new_frequency = frequency + 1
        self._key_to_frequency[key] = new_frequency
        self._frequency_to_keys.setdefault(new_frequency, OrderedDict())[key] = None

    def _remove_key(self, key: ObjectKey) -> None:
        frequency = self._key_to_frequency.pop(key, None)
        if frequency is None:
            return
        keys = self._frequency_to_keys[frequency]
        del keys[key]
        if not keys:
            del self._frequency_to_keys[frequency]
