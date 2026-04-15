# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified touch feature in end_session.

This module tests:
1. L1EvictionPolicy.on_l1_keys_accessed bridges to policy.on_keys_touched
2. L1EvictionPolicy.on_l1_keys_read_finished also triggers touch
3. L1Manager.touch_keys dispatches to all registered listeners
4. MPCacheEngine.end_session performs unified touch with correct keys
5. StorageManager.touch_l1_keys delegates to L1Manager
6. L1Manager.finish_read triggers touch only after all read locks are released
   (non-temporary keys only)
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

# =============================================================================
# Helper Functions
# =============================================================================


def make_key(chunk_hash: int, model: str = "model", kv_rank: int = 0) -> ObjectKey:
    """Create an ObjectKey for testing."""
    hash_bytes = ObjectKey.IntHash2Bytes(chunk_hash)
    return ObjectKey(chunk_hash=hash_bytes, model_name=model, kv_rank=kv_rank)


def make_ipc_key(
    token_ids: list[int],
    chunk_size: int = 4,
    model_name: str = "model",
    world_size: int = 1,
    worker_id: int | None = None,
    request_id: str = "req-1",
) -> IPCCacheEngineKey:
    """Create an IPCCacheEngineKey for testing."""
    return IPCCacheEngineKey.from_token_ids(
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        token_ids=token_ids,
        start=0,
        end=len(token_ids),
        request_id=request_id,
    )


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with small chunk_size for testing."""
    return TokenHasher(chunk_size=4, hash_algorithm="blake3")


# =============================================================================
# L1EvictionPolicy Bridge Tests
# =============================================================================


class TestL1EvictionPolicyAccessedBridge:
    """Tests for L1EvictionPolicy.on_l1_keys_accessed bridging."""

    def test_on_l1_keys_accessed_calls_policy_on_keys_touched(self):
        """on_l1_keys_accessed should delegate to policy.on_keys_touched."""
        mock_policy = MagicMock()
        listener = L1EvictionPolicy(mock_policy)
        keys = [make_key(1), make_key(2)]

        listener.on_l1_keys_accessed(keys)

        mock_policy.on_keys_touched.assert_called_once_with(keys)

    def test_on_l1_keys_read_finished_also_touches(self):
        """on_l1_keys_read_finished should call on_keys_touched."""
        mock_policy = MagicMock()
        listener = L1EvictionPolicy(mock_policy)
        keys = [make_key(1), make_key(2)]

        listener.on_l1_keys_read_finished(keys)

        mock_policy.on_keys_touched.assert_called_once_with(keys)

    def test_on_l1_keys_accessed_with_lru_policy(self):
        """on_l1_keys_accessed should update LRU order via real LRU policy."""
        lru = LRUEvictionPolicy()
        listener = L1EvictionPolicy(lru)

        # Create keys in order: 1, 2, 3
        keys = [make_key(1), make_key(2), make_key(3)]
        lru.on_keys_created(keys)

        # Touch key 1 via the listener bridge
        listener.on_l1_keys_accessed([make_key(1)])

        # Eviction order should now be: 3, 2, 1 (1 is most recent)
        candidates = lru.get_eviction_candidates(3)
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 3
        assert ObjectKey.Bytes2IntHash(candidates[1].chunk_hash) == 2
        assert ObjectKey.Bytes2IntHash(candidates[2].chunk_hash) == 1

    def test_on_l1_keys_read_finished_updates_lru(self):
        """on_l1_keys_read_finished should update LRU order."""
        lru = LRUEvictionPolicy()
        listener = L1EvictionPolicy(lru)

        # Create keys in order: 1, 2, 3
        keys = [make_key(1), make_key(2), make_key(3)]
        lru.on_keys_created(keys)

        # Call on_l1_keys_read_finished on key 1 - should touch it
        listener.on_l1_keys_read_finished([make_key(1)])

        # Key 1 is now most recently touched, eviction order: 2, 3, 1
        candidates = lru.get_eviction_candidates(3)
        assert ObjectKey.Bytes2IntHash(candidates[0].chunk_hash) == 3
        assert ObjectKey.Bytes2IntHash(candidates[1].chunk_hash) == 2
        assert ObjectKey.Bytes2IntHash(candidates[2].chunk_hash) == 1


class TestL1EvictionPolicyUnifiedTouchOrder:
    """Tests for the unified touch producing correct eviction order.

    The key benefit of unified touch is that all chunks of a request
    are touched in order, so eviction proceeds from the last chunk
    backwards (most recent chunk evicted last).
    """

    def test_unified_touch_produces_reverse_eviction_order(self):
        """Touching keys in order [1,2,3] should evict 1 first (LRU)."""
        lru = LRUEvictionPolicy()
        listener = L1EvictionPolicy(lru)

        keys = [make_key(i) for i in range(1, 6)]
        lru.on_keys_created(keys)

        # Unified touch: touch all keys in order (simulating end_session)
        listener.on_l1_keys_accessed(keys)

        # Eviction should be in reverse order: 1 first (least recently touched)
        # because on_keys_touched processes them sequentially,
        # so key 5 is the most recently touched
        candidates = lru.get_eviction_candidates(5)
        evicted_ids = [ObjectKey.Bytes2IntHash(c.chunk_hash) for c in candidates]
        # LRU evicts from the back of the ordered dict (least recent first)
        # After touching [1,2,3,4,5] in order, 1 is least recent, 5 is most recent
        assert evicted_ids == [5, 4, 3, 2, 1]

    def test_per_read_touch_vs_unified_touch_order(self):
        """Compare per-read touch (old behavior) vs unified touch (new behavior).

        Old behavior: each retrieve touches its keys immediately.
        New behavior: all keys are touched together at end_session.

        With unified touch, the eviction order reflects the full request's
        chunk ordering, not the order of individual retrieve calls.
        """
        # Scenario: request has chunks [1,2,3,4,5]
        # Retrieve touches [1,2,3] (prefix hit), Store creates [4,5]
        lru = LRUEvictionPolicy()

        keys = [make_key(i) for i in range(1, 6)]
        lru.on_keys_created(keys)

        # Unified touch at end_session: touch all in order
        lru.on_keys_touched(keys)

        candidates = lru.get_eviction_candidates(5)
        evicted_ids = [ObjectKey.Bytes2IntHash(c.chunk_hash) for c in candidates]
        # Last chunk (5) should be evicted last (most recently touched)
        assert evicted_ids[-1] == 1


# =============================================================================
# End Session Touch Integration Tests
# =============================================================================


class TestEndSessionTouchKeys:
    """Tests for the end_session unified touch key generation logic.

    These tests verify that end_session correctly:
    1. Retrieves the session with accumulated hashes
    2. Generates the correct ObjectKeys using lookup_ipc_key
    3. Handles edge cases (no session, no lookup key, empty hashes)
    """

    def test_end_session_generates_correct_keys(self, hasher: TokenHasher):
        """end_session should generate ObjectKeys from session hashes and lookup key."""
        mgr = SessionManager(hasher, ttl=600)
        session = mgr.get_or_create("req-1")

        tokens = list(range(12))  # 3 chunks of 4
        session.set_tokens(tokens)
        session.get_hashes(0, 12)

        ipc_key = make_ipc_key(tokens, chunk_size=4, worker_id=None, request_id="req-1")
        session.lookup_ipc_key = ipc_key

        # Simulate end_session logic
        removed = mgr.remove("req-1")
        assert removed is not None
        assert removed.lookup_ipc_key is not None

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in removed.get_hashes(0)]
        obj_keys = ipc_key_to_object_keys(removed.lookup_ipc_key, chunk_hashes)

        # With world_size=1 and worker_id=None, should have 3 keys
        assert len(obj_keys) == 3

    def test_end_session_expands_keys_for_world_size(self, hasher: TokenHasher):
        """end_session should expand keys for all workers when world_size > 1."""
        mgr = SessionManager(hasher, ttl=600)
        session = mgr.get_or_create("req-1")

        tokens = list(range(8))  # 2 chunks of 4
        session.set_tokens(tokens)
        session.get_hashes(0, 8)

        ipc_key = make_ipc_key(
            tokens,
            chunk_size=4,
            world_size=2,
            worker_id=None,
            request_id="req-1",
        )
        session.lookup_ipc_key = ipc_key

        removed = mgr.remove("req-1")
        assert removed is not None
        assert removed.lookup_ipc_key is not None

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in removed.get_hashes(0)]
        obj_keys = ipc_key_to_object_keys(removed.lookup_ipc_key, chunk_hashes)

        # 2 chunks * 2 workers = 4 keys
        assert len(obj_keys) == 4

    def test_end_session_no_session_skips_touch(self, hasher: TokenHasher):
        """end_session should skip touch when session doesn't exist."""
        mgr = SessionManager(hasher, ttl=600)
        removed = mgr.remove("nonexistent")
        assert removed is None
        # No error should occur

    def test_end_session_no_lookup_key_skips_touch(self, hasher: TokenHasher):
        """end_session should skip touch when session has no lookup_ipc_key."""
        mgr = SessionManager(hasher, ttl=600)
        session = mgr.get_or_create("req-1")
        session.set_tokens(list(range(8)))
        session.get_hashes(0, 8)
        # Don't set lookup_ipc_key

        removed = mgr.remove("req-1")
        assert removed is not None
        assert removed.lookup_ipc_key is None
        # No error should occur

    def test_end_session_empty_hashes_produces_no_keys(self, hasher: TokenHasher):
        """end_session should produce no keys when session has no computed hashes."""
        mgr = SessionManager(hasher, ttl=600)
        session = mgr.get_or_create("req-1")

        ipc_key = make_ipc_key(
            list(range(8)), chunk_size=4, worker_id=None, request_id="req-1"
        )
        session.lookup_ipc_key = ipc_key
        # Don't compute any hashes

        removed = mgr.remove("req-1")
        assert removed is not None
        assert removed.lookup_ipc_key is not None

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in removed.get_hashes(0)]
        obj_keys = ipc_key_to_object_keys(removed.lookup_ipc_key, chunk_hashes)
        assert len(obj_keys) == 0

    def test_end_session_hashes_cover_retrieve_and_store(self, hasher: TokenHasher):
        """Session hashes should cover both retrieve and store ranges.

        Simulates the typical flow:
        1. lookup computes all hashes (but via TokenHasher, not Session)
        2. retrieve calls session.get_hashes(0, hit_end)
        3. store calls session.get_hashes(hit_end, total)
        4. end_session uses session.get_hashes(0) to get all
        """
        mgr = SessionManager(hasher, ttl=600)
        session = mgr.get_or_create("req-1")

        tokens = list(range(20))  # 5 chunks of 4
        session.set_tokens(tokens)

        # Simulate retrieve: first 3 chunks
        retrieve_hashes = session.get_hashes(0, 12)
        assert len(retrieve_hashes) == 3

        # Simulate store: last 2 chunks
        store_hashes = session.get_hashes(12, 20)
        assert len(store_hashes) == 2

        # All hashes should cover all 5 chunks
        all_hashes = session.get_hashes(0)
        assert len(all_hashes) == 5
        assert all_hashes == retrieve_hashes + store_hashes


# =============================================================================
# L1Manager finish_read Touch Behaviour Tests
# =============================================================================


# Guard: skip if L1Manager cannot be imported (e.g. missing native deps)
_l1_manager_available = True
try:
    # Third Party
    import torch

    if not torch.cuda.is_available():
        _l1_manager_available = False
    else:
        # First Party
        from lmcache.v1.distributed.config import (
            L1ManagerConfig,
            L1MemoryManagerConfig,
        )
        from lmcache.v1.distributed.l1_manager import L1Manager
except ImportError:
    _l1_manager_available = False


def _should_use_lazy_alloc() -> bool:
    """Determine if lazy allocation should be used."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture
def l1_manager():
    """Create a basic L1Manager for touch-related tests."""
    mem_cfg = L1MemoryManagerConfig(
        size_in_bytes=128 * 1024 * 1024,
        use_lazy=_should_use_lazy_alloc(),
        init_size_in_bytes=64 * 1024 * 1024,
        align_bytes=0x1000,
    )
    cfg = L1ManagerConfig(
        memory_config=mem_cfg,
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(cfg)
    yield mgr
    mgr.close()


@pytest.fixture
def l1_layout():
    """Create a basic MemoryLayoutDesc for touch-related tests."""
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def _make_ready(
    manager: "L1Manager",
    key: ObjectKey,
    layout,
    is_temporary: bool = False,
):
    """Helper: create a ready (write-finished) object in L1Manager."""
    manager.reserve_write([key], [is_temporary], layout)
    manager.finish_write([key])


@pytest.mark.skipif(
    not _l1_manager_available,
    reason="L1Manager not available (CUDA or native deps missing)",
)
class TestFinishReadTouchBehaviour:
    """Tests that L1Manager.finish_read triggers touch only after
    all read locks are released.

    The new behaviour:
    - on_l1_keys_read_finished is called with keys whose read_lock
      count has dropped to zero AND are non-temporary.
    - Keys that still have remaining read locks are NOT touched.
    - Temporary keys are NOT touched (they are deleted instead).
    """

    def test_single_read_lock_triggers_touch_on_release(self, l1_manager, l1_layout):
        """A single reserve_read + finish_read should trigger touch."""
        mock_listener = MagicMock(spec=L1EvictionPolicy)
        l1_manager.register_listener(mock_listener)

        key = make_key(100)
        _make_ready(l1_manager, key, l1_layout)

        l1_manager.reserve_read([key])
        l1_manager.finish_read([key])

        # on_l1_keys_read_finished should be called with [key]
        mock_listener.on_l1_keys_read_finished.assert_called()
        touched = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key in touched

    def test_multiple_read_locks_no_touch_until_all_released(
        self, l1_manager, l1_layout
    ):
        """With multiple read locks, touch should NOT fire until
        the last lock is released."""
        mock_listener = MagicMock(spec=L1EvictionPolicy)
        l1_manager.register_listener(mock_listener)

        key = make_key(200)
        _make_ready(l1_manager, key, l1_layout)

        # Acquire 3 read locks
        l1_manager.reserve_read([key])
        l1_manager.reserve_read([key])
        l1_manager.reserve_read([key])

        # Release first lock — key still has 2 remaining
        l1_manager.finish_read([key])
        touched_1 = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key not in touched_1, "Key should NOT be touched while read locks remain"

        # Release second lock — key still has 1 remaining
        mock_listener.reset_mock()
        l1_manager.finish_read([key])
        touched_2 = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key not in touched_2, "Key should NOT be touched while read locks remain"

        # Release last lock — now touch should fire
        mock_listener.reset_mock()
        l1_manager.finish_read([key])
        touched_3 = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key in touched_3, (
            "Key should be touched after all read locks are released"
        )

    def test_extra_count_all_released_triggers_touch(self, l1_manager, l1_layout):
        """reserve_read(extra_count=2) + finish_read(extra_count=2)
        releases all 3 locks at once and should trigger touch."""
        mock_listener = MagicMock(spec=L1EvictionPolicy)
        l1_manager.register_listener(mock_listener)

        key = make_key(300)
        _make_ready(l1_manager, key, l1_layout)

        l1_manager.reserve_read([key], extra_count=2)
        l1_manager.finish_read([key], extra_count=2)

        touched = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key in touched

    def test_extra_count_partial_release_no_touch(self, l1_manager, l1_layout):
        """Partial release via extra_count should NOT trigger touch."""
        mock_listener = MagicMock(spec=L1EvictionPolicy)
        l1_manager.register_listener(mock_listener)

        key = make_key(400)
        _make_ready(l1_manager, key, l1_layout)

        # Acquire 3 locks (1 + extra_count=2)
        l1_manager.reserve_read([key], extra_count=2)

        # Release 2 of 3 (1 + extra_count=1) — 1 lock remains
        l1_manager.finish_read([key], extra_count=1)
        touched = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key not in touched, "Key should NOT be touched while read locks remain"

        # Release last lock
        mock_listener.reset_mock()
        l1_manager.finish_read([key])
        touched_final = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key in touched_final

    def test_temporary_key_not_touched_on_release(self, l1_manager, l1_layout):
        """Temporary keys should NOT be touched — they are deleted instead."""
        mock_listener = MagicMock(spec=L1EvictionPolicy)
        l1_manager.register_listener(mock_listener)

        key = make_key(500)
        _make_ready(l1_manager, key, l1_layout, is_temporary=True)

        l1_manager.reserve_read([key])
        l1_manager.finish_read([key])

        # Key should be deleted, not touched
        touched = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        assert key not in touched, "Temporary key should NOT be touched"
        # Verify it was deleted via on_l1_keys_deleted_by_manager
        mock_listener.on_l1_keys_deleted_by_manager.assert_called()
        deleted = mock_listener.on_l1_keys_deleted_by_manager.call_args[0][0]
        assert key in deleted

    def test_mixed_keys_only_fully_released_non_temp_touched(
        self, l1_manager, l1_layout
    ):
        """With a mix of keys, only fully-released non-temporary keys
        should be touched."""
        mock_listener = MagicMock(spec=L1EvictionPolicy)
        l1_manager.register_listener(mock_listener)

        key_normal = make_key(601)
        key_still_locked = make_key(602)
        key_temp = make_key(603)

        _make_ready(l1_manager, key_normal, l1_layout)
        _make_ready(l1_manager, key_still_locked, l1_layout)
        _make_ready(l1_manager, key_temp, l1_layout, is_temporary=True)

        # key_normal: 1 read lock
        l1_manager.reserve_read([key_normal])
        # key_still_locked: 2 read locks (will only release 1)
        l1_manager.reserve_read([key_still_locked])
        l1_manager.reserve_read([key_still_locked])
        # key_temp: 1 read lock
        l1_manager.reserve_read([key_temp])

        # Finish read for all three (releasing 1 lock each)
        l1_manager.finish_read([key_normal, key_still_locked, key_temp])

        touched = mock_listener.on_l1_keys_read_finished.call_args[0][0]
        # Only key_normal should be touched
        assert key_normal in touched
        assert key_still_locked not in touched, (
            "Key with remaining read locks should NOT be touched"
        )
        assert key_temp not in touched, "Temporary key should NOT be touched"


# =============================================================================
# StoreListener No-op Tests
# =============================================================================


class TestStoreListenerAccessedNoop:
    """Tests that StoreListener.on_l1_keys_accessed is a no-op."""

    def test_store_listener_accessed_is_noop(self):
        """StoreListener.on_l1_keys_accessed should not raise."""
        # Import here to avoid import errors if store_controller has
        # heavy dependencies
        # First Party
        from lmcache.v1.distributed.storage_controllers.store_controller import (
            StoreListener,
        )

        # StoreListener requires an event_fd; use a mock
        listener = MagicMock(spec=StoreListener)
        # Call the real method on the class
        StoreListener.on_l1_keys_accessed(listener, [make_key(1)])
        # Should not raise - that's the test
