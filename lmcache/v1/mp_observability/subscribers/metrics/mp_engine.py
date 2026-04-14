# SPDX-License-Identifier: Apache-2.0

"""MP engine metrics subscriber — OTel counters and histograms
for store, retrieve, and lookup operations."""

# Future
from __future__ import annotations

# Standard
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventCallback,
    EventSubscriber,
)

logger = init_logger(__name__)


class MPEngineMetricsSubscriber(EventSubscriber):
    """Counters and latency histograms for MP engine operations.

    Counters:
    - ``lmcache_mp.store_requests``       — store operations
    - ``lmcache_mp.store_chunks``         — chunks stored
    - ``lmcache_mp.retrieve_requests``    — retrieve operations
    - ``lmcache_mp.retrieve_chunks``      — chunks retrieved
    - ``lmcache_mp.lookup_requests``      — lookup/prefetch ops
    - ``lmcache_mp.lookup_hit_chunks``    — chunks found in lookup
    - ``lmcache_mp.lookup_miss_requests`` — lookups with 0 hits
    - ``lmcache_mp.end_session_requests`` — sessions ended

    Histograms (seconds):
    - ``lmcache_mp.store_duration_seconds``
    - ``lmcache_mp.retrieve_duration_seconds``
    - ``lmcache_mp.lookup_duration_seconds``
    """

    # Default TTL for orphaned START timestamps (seconds)
    STALE_TTL: float = 300.0  # 5 minutes

    def __init__(
        self,
        stale_ttl: float = STALE_TTL,
    ) -> None:
        self._stale_ttl = stale_ttl
        meter = metrics.get_meter("lmcache.mp_engine")

        # Store counters
        self._store_requests = meter.create_counter(
            "lmcache_mp.store_requests",
            description="Total GPU->CPU store operations",
        )
        self._store_chunks = meter.create_counter(
            "lmcache_mp.store_chunks",
            description="Total chunks stored (GPU->CPU)",
        )

        # Retrieve counters
        self._retrieve_requests = meter.create_counter(
            "lmcache_mp.retrieve_requests",
            description="Total CPU->GPU retrieve operations",
        )
        self._retrieve_chunks = meter.create_counter(
            "lmcache_mp.retrieve_chunks",
            description="Total chunks retrieved (CPU->GPU)",
        )

        # Lookup counters
        self._lookup_requests = meter.create_counter(
            "lmcache_mp.lookup_requests",
            description="Total lookup/prefetch operations",
        )
        self._lookup_hit_chunks = meter.create_counter(
            "lmcache_mp.lookup_hit_chunks",
            description="Total chunks found during lookup",
        )
        self._lookup_miss_requests = meter.create_counter(
            "lmcache_mp.lookup_miss_requests",
            description="Total lookups with zero hits",
        )

        # Session counter
        self._end_session_requests = meter.create_counter(
            "lmcache_mp.end_session_requests",
            description="Total sessions ended",
        )

        # Latency histograms
        self._store_duration = meter.create_histogram(
            "lmcache_mp.store_duration_seconds",
            description="Store operation latency (seconds)",
            unit="s",
        )
        self._retrieve_duration = meter.create_histogram(
            "lmcache_mp.retrieve_duration_seconds",
            description="Retrieve operation latency (seconds)",
            unit="s",
        )
        self._lookup_duration = meter.create_histogram(
            "lmcache_mp.lookup_duration_seconds",
            description=("Lookup/prefetch end-to-end latency (seconds)"),
            unit="s",
        )

        # Orphaned start cleanup counter
        self._orphaned_cleaned = meter.create_counter(
            "lmcache_mp.orphaned_starts_cleaned",
            description=("Orphaned START timestamps cleaned by TTL scan"),
        )

        # Pending START timestamps keyed by session_id
        self._store_starts: dict[str, float] = {}
        self._retrieve_starts: dict[str, float] = {}
        self._lookup_starts: dict[str, float] = {}

    def get_subscriptions(
        self,
    ) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            EventType.MP_RETRIEVE_START: (self._on_retrieve_start),
            EventType.MP_RETRIEVE_END: self._on_retrieve_end,
            EventType.MP_LOOKUP_PREFETCH_START: (self._on_lookup_start),
            EventType.MP_LOOKUP_PREFETCH_END: (self._on_lookup_end),
            EventType.MP_VLLM_END_SESSION: (self._on_end_session),
        }

    # -- Store -------------------------------------------------------------

    def _on_store_start(self, event: Event) -> None:
        self._store_starts[event.session_id] = event.timestamp

    def _on_store_end(self, event: Event) -> None:
        self._store_requests.add(1)
        stored = event.metadata.get("stored_count", 0)
        self._store_chunks.add(stored)

        start_ts = self._store_starts.pop(event.session_id, None)
        if start_ts is not None:
            duration = event.timestamp - start_ts
            self._store_duration.record(duration)

    # -- Retrieve ----------------------------------------------------------

    def _on_retrieve_start(self, event: Event) -> None:
        self._retrieve_starts[event.session_id] = event.timestamp

    def _on_retrieve_end(self, event: Event) -> None:
        self._retrieve_requests.add(1)
        retrieved = event.metadata.get("retrieved_count", 0)
        self._retrieve_chunks.add(retrieved)

        start_ts = self._retrieve_starts.pop(event.session_id, None)
        if start_ts is not None:
            duration = event.timestamp - start_ts
            self._retrieve_duration.record(duration)

    # -- Lookup/Prefetch ---------------------------------------------------

    def _on_lookup_start(self, event: Event) -> None:
        self._lookup_starts[event.session_id] = event.timestamp

    def _on_lookup_end(self, event: Event) -> None:
        self._lookup_requests.add(1)
        found = event.metadata.get("found_count", 0)
        self._lookup_hit_chunks.add(found)
        if found == 0:
            self._lookup_miss_requests.add(1)

        start_ts = self._lookup_starts.pop(event.session_id, None)
        if start_ts is not None:
            duration = event.timestamp - start_ts
            self._lookup_duration.record(duration)

    # -- Session -----------------------------------------------------------

    def _on_end_session(self, event: Event) -> None:
        self._end_session_requests.add(1)
        # Clean up any orphaned START timestamps to prevent
        # memory leaks when END events are never delivered.
        sid = event.session_id
        self._store_starts.pop(sid, None)
        self._retrieve_starts.pop(sid, None)
        self._lookup_starts.pop(sid, None)

    # -- TTL cleanup -------------------------------------------------------

    def cleanup_stale(self) -> int:
        """Remove START timestamps older than *stale_ttl*.

        Returns:
            Total number of orphaned entries removed.
        """
        now = time.monotonic()
        cutoff = now - self._stale_ttl
        cleaned = 0
        for name, starts in (
            ("store", self._store_starts),
            ("retrieve", self._retrieve_starts),
            ("lookup", self._lookup_starts),
        ):
            stale_ids = [sid for sid, ts in starts.items() if ts < cutoff]
            for sid in stale_ids:
                starts.pop(sid, None)
                cleaned += 1
                logger.warning(
                    "Cleaned orphaned %s START for session %s (age=%.0fs)",
                    name,
                    sid,
                    now - cutoff,
                )
        if cleaned:
            self._orphaned_cleaned.add(cleaned)
        return cleaned
