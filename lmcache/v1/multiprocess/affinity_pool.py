# SPDX-License-Identifier: Apache-2.0
"""
Thread pool that keeps one client's affinity handlers off other threads.

Tasks sharing an ``affinity_key`` are bound to one worker, so they never run
at the same time. That mutual exclusion is the point: GPU-bound handlers
(STORE / RETRIEVE) reuse a per-client temporary GPU buffer, and serializing
them on one thread removes the need for locks on that buffer (and for the
deadlock-prone CUDA driver lock such locks had to coordinate with).

The ``affinity_key`` identifies a connection, not a vLLM instance: ZMQ uses
the client identity and gRPC uses the client-id metadata or peer address.
Reconnecting may therefore change the key. When distinct keys outnumber
workers, some share a worker and their tasks are still serialized.

Each worker drains a ``queue.Queue`` in FIFO order, but that ordering is an
implementation detail, not a contract. Affinity provides mutual exclusion,
not per-client request ordering; cross-request causality comes from the
client's request/response round trips and from state keyed by request id.
"""

# Standard
from concurrent.futures import Future
import queue
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Sentinel object to signal worker shutdown
_SHUTDOWN = object()


class AffinityThreadPool:
    """Route tasks by key so each client's handlers never run concurrently.

    Binding a key to one worker is a mutual-exclusion mechanism, not a
    request-ordering guarantee. The FIFO behaviour of a worker's queue is an
    implementation detail of ``queue.Queue``; callers must not rely on it for
    correctness. See the module docstring for why the exclusion is needed.

    Not thread-safe: the request transport must serialize calls to ``submit()``.
    ZMQ submits from its main loop, while gRPC protects submission with a lock.

    Args:
        max_workers: Number of worker threads.
        thread_name_prefix: Prefix for worker thread names.
    """

    def __init__(
        self,
        max_workers: int,
        thread_name_prefix: str = "affinity",
    ) -> None:
        self._num_workers = max_workers
        self._queues: list[queue.Queue] = [queue.Queue() for _ in range(max_workers)]
        self._threads: list[threading.Thread] = []
        # Maps an affinity_key -> the worker slot (thread index) bound to it.
        self._key_to_slot: dict[int, int] = {}
        self._next_slot = 0
        self._overflow_warned = False
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self._queues[i],),
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "Created AffinityThreadPool '%s' with %d worker slots: up to %d "
            "distinct affinity keys each bind to their own thread before slots "
            "are shared. Compare this against the number of clients expected to "
            "connect to confirm routing.",
            thread_name_prefix,
            max_workers,
            max_workers,
        )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    @staticmethod
    def _worker(q: queue.Queue) -> None:
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                break
            future, fn, args, kwargs = item
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except BaseException as exc:
                    future.set_exception(exc)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _slot_for_key(self, affinity_key: int) -> int:
        """Return the worker slot bound to ``affinity_key``, assigning on first use.

        Returns:
            The worker slot (an index in ``[0, _num_workers)``) for the key.
        """
        slot = self._key_to_slot.get(affinity_key)
        if slot is not None:
            return slot

        # First time we see this key -- bind it to the next free slot.
        slot = self._next_slot % self._num_workers
        is_overflow = self._next_slot >= self._num_workers
        self._key_to_slot[affinity_key] = slot
        self._next_slot += 1

        logger.info(
            "AffinityThreadPool: affinity_key=%d assigned to worker "
            "slot %d of %d (thread %s); %d distinct key(s) now bound",
            affinity_key,
            slot,
            self._num_workers,
            self._threads[slot].name,
            len(self._key_to_slot),
        )
        if is_overflow and not self._overflow_warned:
            self._overflow_warned = True
            logger.warning(
                "AffinityThreadPool: affinity_key=%d wrapped onto worker slot "
                "%d (only %d workers), so it shares a thread with an earlier "
                "key and their tasks are serialized. Increase the worker count "
                "to give each client its own thread.",
                affinity_key,
                slot,
                self._num_workers,
            )
        return slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn, *args, affinity_key: int = 0, **kwargs) -> Future:
        """Submit *fn* for execution on the worker bound to *affinity_key*.

        Returns a :class:`concurrent.futures.Future`.
        """
        future: Future = Future()
        slot = self._slot_for_key(affinity_key)
        self._queues[slot].put((future, fn, args, kwargs))
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool.

        Sends a shutdown sentinel to every worker.  If *wait* is true, blocks
        until all workers have exited.
        """
        for q in self._queues:
            q.put(_SHUTDOWN)
        if wait:
            for t in self._threads:
                t.join()
