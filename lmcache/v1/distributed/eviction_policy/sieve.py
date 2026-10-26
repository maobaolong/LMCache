# SPDX-License-Identifier: Apache-2.0
"""SIEVE eviction policy for multiprocess cache tiers."""

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


class SieveEvictionPolicy(EvictionPolicy):
    """Use a persistent hand and one visited bit to select eviction victims.

    Keys retain FIFO insertion order. Accesses only set a visited bit instead
    of moving an entry, while each eviction cycle advances a persistent hand.
    A visited entry receives one second chance: the scan clears its bit and
    continues. This makes the policy resistant to one-time scans and keeps the
    access path cheaper than an ordered LRU update.

    Keys within one request are inserted in reverse order so suffix chunks are
    considered before prefix chunks, matching the other MP eviction policies.

    Thread Safety:
        Every state transition is protected by one lock.
    """

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ) -> None:
        """Initialize an empty SIEVE policy.

        Args:
            default_destination: Destination used when no destination has been
                registered by an eviction controller.
        """
        self._lock = threading.Lock()
        self._queue: OrderedDict[ObjectKey, bool] = OrderedDict()
        self._hand: ObjectKey | None = None
        self._pending_admissions: set[ObjectKey] = set()
        self._pending_evictions: set[ObjectKey] = set()
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

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        """Track newly resident keys without marking them visited.

        A completion notification for a previously reserved key preserves its
        existing queue position and visited state.

        Args:
            keys: Keys that became resident after a completed write.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                self._pending_evictions.discard(key)
                if key in self._pending_admissions:
                    self._pending_admissions.remove(key)
                    continue
                self._insert(key)

    def on_keys_reserved(self, keys: list[ObjectKey]) -> None:
        """Track write reservations as cold entries eligible for eviction.

        Args:
            keys: Keys reserved for creation.
        """
        if not keys:
            return
        with self._lock:
            for key in reversed(keys):
                self._pending_evictions.discard(key)
                self._pending_admissions.add(key)
                self._insert(key)

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        """Give resident keys one second chance without reordering them.

        Args:
            keys: Resident keys that were accessed.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                if key in self._queue:
                    self._queue[key] = True

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        """Forget keys after the cache confirms their removal.

        Args:
            keys: Keys confirmed deleted by the L1 manager or L2 adapter.
        """
        if not keys:
            return
        with self._lock:
            for key in keys:
                self._remove(key)
                self._pending_admissions.discard(key)
                self._pending_evictions.discard(key)

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter: Callable[[ObjectKey], bool] | None = None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        """Choose cold entries by advancing the persistent SIEVE hand.

        Args:
            expected_ratio: Approximate fraction of resident keys to evict,
                clamped to ``[0.0, 1.0]``.
            key_eligible_filter: Optional predicate used to skip locked or
                otherwise ineligible keys.
            cache_salt: Ignored because SIEVE is a global, non-isolated policy.

        Returns:
            Zero or one eviction action containing eligible keys in SIEVE scan
            order. A positive ratio selects at least one key when possible.
        """
        del cache_salt
        with self._lock:
            resident_size = len(self._queue)
            if resident_size == 0:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target_count = int(resident_size * expected_ratio)
            if expected_ratio > 0 and target_count == 0:
                target_count = 1
            if target_count == 0:
                return []

            ordered_keys = tuple(self._queue)
            start_index = self._hand_index(ordered_keys)
            scan_index = start_index
            scan_budget = 2 * resident_size
            victims: list[ObjectKey] = []
            selected: set[ObjectKey] = set()

            for _ in range(scan_budget):
                key = ordered_keys[scan_index]
                scan_index = (scan_index + 1) % resident_size
                self._hand = ordered_keys[scan_index]

                if key in selected or key in self._pending_evictions:
                    continue
                if key_eligible_filter is not None and not key_eligible_filter(key):
                    continue
                if self._queue[key]:
                    self._queue[key] = False
                    continue

                selected.add(key)
                victims.append(key)
                if len(victims) >= target_count:
                    break

            if not victims:
                return []

            self._pending_evictions.update(victims)
            destination = (
                self._destinations[0]
                if self._destinations
                else self._default_destination
            )
            return [EvictionAction(keys=victims, destination=destination)]

    def get_num_tracked_keys(self) -> int:
        """Return the number of resident keys tracked by the policy."""
        with self._lock:
            return len(self._queue)

    def _insert(self, key: ObjectKey) -> None:
        self._remove(key)
        self._queue[key] = False
        if self._hand is None:
            self._hand = key

    def _remove(self, key: ObjectKey) -> None:
        if key not in self._queue:
            return
        if self._hand == key:
            ordered_keys = tuple(self._queue)
            key_index = ordered_keys.index(key)
            self._hand = (
                ordered_keys[(key_index + 1) % len(ordered_keys)]
                if len(ordered_keys) > 1
                else None
            )
        del self._queue[key]

    def _hand_index(self, ordered_keys: tuple[ObjectKey, ...]) -> int:
        if self._hand is None or self._hand not in self._queue:
            self._hand = ordered_keys[0]
            return 0
        return ordered_keys.index(self._hand)
