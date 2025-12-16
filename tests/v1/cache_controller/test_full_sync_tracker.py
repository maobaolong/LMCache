# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FullSyncTracker."""

# Standard
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.full_sync_tracker import FullSyncTracker
from lmcache.v1.cache_controller.utils import (
    FullSyncState,
    RegistryTree,
    WorkerSyncInfo,
)


class TestHelper:
    """Helper methods for FullSyncTracker tests."""

    @staticmethod
    def create_tracker(
        completion_threshold=0.8, sync_timeout_s=300.0
    ) -> tuple[FullSyncTracker, RegistryTree]:
        """Create a FullSyncTracker with optional parameters.

        Returns:
            tuple[FullSyncTracker, RegistryTree]
        """
        registry_tree = RegistryTree()
        tracker = FullSyncTracker(
            registry_tree=registry_tree,
            completion_threshold=completion_threshold,
            sync_timeout_s=sync_timeout_s,
        )
        return tracker, registry_tree

    @staticmethod
    def register_worker(
        registry_tree, instance_id, worker_id, ip="192.168.1.1", port=8000
    ):
        """Register a worker in RegistryTree."""
        registry_tree.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

    @staticmethod
    def start_and_complete_sync(
        tracker, instance_id, worker_id, sync_id, total_keys=100, batch_count=5
    ):
        """Helper to start and complete a sync."""
        tracker.start_sync(instance_id, worker_id, sync_id, total_keys, batch_count)
        tracker.complete_sync(instance_id, worker_id, sync_id, total_keys)

    @staticmethod
    def create_worker_sync_info(
        sync_id="test_sync",
        state=FullSyncState.SYNCING,
        start_time=1000.0,
        total_keys=100,
        batch_count=5,
    ):
        """Create a WorkerSyncInfo for testing."""
        return WorkerSyncInfo(
            sync_id=sync_id,
            state=state,
            start_time=start_time,
            expected_total_keys=total_keys,
            expected_batch_count=batch_count,
        )


class TestWorkerSyncInfo:
    """Test cases for WorkerSyncInfo."""

    @pytest.mark.parametrize(
        "sync_id,state,start_time,total_keys,batch_count,expected",
        [
            (
                "sync_12345",
                FullSyncState.SYNCING,
                1000.0,
                100,
                5,
                {
                    "sync_id": "sync_12345",
                    "state": FullSyncState.SYNCING,
                    "start_time": 1000.0,
                    "last_activity_time": 1000.0,
                },
            ),
            (
                "sync_test",
                FullSyncState.COMPLETED,
                2000.0,
                500,
                10,
                {
                    "sync_id": "sync_test",
                    "state": FullSyncState.COMPLETED,
                    "start_time": 2000.0,
                    "last_activity_time": 2000.0,
                },
            ),
        ],
    )
    def test_worker_sync_info_creation(
        self, sync_id, state, start_time, total_keys, batch_count, expected
    ):
        """Test WorkerSyncInfo creation and attributes."""
        info = WorkerSyncInfo(
            sync_id=sync_id,
            state=state,
            start_time=start_time,
            expected_total_keys=total_keys,
            expected_batch_count=batch_count,
        )

        assert info.sync_id == expected["sync_id"]
        assert info.state == expected["state"]
        assert info.start_time == expected["start_time"]
        assert info.last_activity_time == expected["last_activity_time"]
        assert info.expected_total_keys == total_keys
        assert info.expected_batch_count == batch_count
        assert info.received_batches == set()
        assert info.received_keys_count == 0

    def test_worker_sync_info_last_activity_time(self):
        """Test that last_activity_time is auto-set from start_time."""
        info = WorkerSyncInfo(
            sync_id="test_sync",
            state=FullSyncState.SYNCING,
            start_time=2500.0,
            expected_total_keys=50,
            expected_batch_count=2,
        )
        assert info.last_activity_time == 2500.0


class TestFullSyncTracker:
    """Test cases for FullSyncTracker."""

    @pytest.mark.parametrize(
        "completion_threshold,sync_timeout_s,expected_threshold,expected_timeout",
        [
            (0.8, 300.0, 0.8, 300.0),
            (0.9, 600.0, 0.9, 600.0),
            (0.5, 150.0, 0.5, 150.0),
        ],
    )
    def test_init_defaults_and_custom(
        self, completion_threshold, sync_timeout_s, expected_threshold, expected_timeout
    ):
        """Test FullSyncTracker initialization with default and custom parameters."""
        tracker, _ = TestHelper.create_tracker(completion_threshold, sync_timeout_s)

        assert tracker.completion_threshold == expected_threshold
        assert tracker.sync_timeout_s == expected_timeout
        assert tracker._need_full_sync_all is True

    def test_set_need_full_sync_all(self):
        """Test setting the global full sync flag."""
        tracker, _ = TestHelper.create_tracker()

        tracker.set_need_full_sync_all(False)
        assert tracker._need_full_sync_all is False

        tracker.set_need_full_sync_all(True)
        assert tracker._need_full_sync_all is True

    @pytest.mark.parametrize(
        "worker_sync_state,need_full_sync_reason",
        [
            ("not_syncing", "controller_restart"),
            ("syncing", None),
            ("completed", None),
            ("failed", "sync_failed_retry"),
        ],
    )
    def test_should_request_full_sync(self, worker_sync_state, need_full_sync_reason):
        """Test should_request_full_sync under different sync states."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)

        # Setup worker state based on parameter
        if worker_sync_state == "syncing":
            tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
        elif worker_sync_state == "completed":
            TestHelper.start_and_complete_sync(tracker, "instance_1", 0, "sync_1")
        elif worker_sync_state == "failed":
            tracker.set_need_full_sync_all(False)  # Disable global flag to test retry
            tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
            tracker.mark_failed("instance_1", 0, "timeout")

        # Query sync requirement
        need_sync, reason = tracker.should_request_full_sync("instance_1", 0)

        if need_full_sync_reason:
            assert need_sync is True
            assert reason == need_full_sync_reason
        else:
            assert need_sync is False
            assert reason is None

    @pytest.mark.parametrize(
        "action,expected_syncing,expected_state",
        [
            ("none", False, None),
            ("start_sync", True, FullSyncState.SYNCING),
            ("complete_sync", False, FullSyncState.COMPLETED),
            ("mark_failed", False, FullSyncState.FAILED),
        ],
    )
    def test_is_worker_syncing(self, action, expected_syncing, expected_state):
        """Test checking if worker is syncing in different states."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)

        # Perform action
        if action == "start_sync":
            tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
        elif action == "complete_sync":
            TestHelper.start_and_complete_sync(tracker, "instance_1", 0, "sync_1")
        elif action == "mark_failed":
            tracker.start_sync("instance_1", 0, "sync_1", 100, 5)
            tracker.mark_failed("instance_1", 0, "test_reason")

        # Check syncing status
        is_syncing = tracker.is_worker_syncing("instance_1", 0)
        assert is_syncing == expected_syncing

        # Check state if action was taken
        if action != "none":
            worker_node = registry_tree.get_worker("instance_1", 0)
            assert worker_node.sync_info.state == expected_state

    @pytest.mark.parametrize(
        "sync_id_conflict,should_succeed",
        [
            (True, False),  # Different sync_id should fail
            (False, True),  # Same sync_id should succeed (retry)
        ],
    )
    def test_start_sync(self, sync_id_conflict, should_succeed):
        """Test starting sync with potential conflict."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)

        # Start first sync
        tracker.start_sync("instance_1", 0, "sync_first", 100, 5)

        # Try to start another sync
        sync_id = "sync_second" if sync_id_conflict else "sync_first"
        result = tracker.start_sync("instance_1", 0, sync_id, 200, 10)

        assert result == should_succeed

        # Verify sync info if successful
        if should_succeed:
            worker_node = registry_tree.get_worker("instance_1", 0)
            sync_info = worker_node.sync_info
            assert sync_info.sync_id == sync_id
            assert sync_info.state == FullSyncState.SYNCING
            assert sync_info.expected_total_keys == (100 if sync_id_conflict else 200)

    @pytest.mark.parametrize(
        "batch_count,keys_per_batch,expected_keys_count",
        [
            (5, 20, 100),
            (2, 50, 100),
            (1, 100, 100),
        ],
    )
    def test_receive_batches(self, batch_count, keys_per_batch, expected_keys_count):
        """Test receiving multiple batches."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, batch_count)

        # Send batches
        for batch_id in range(batch_count):
            result = tracker.receive_batch(
                "instance_1", 0, "sync_1", batch_id, keys_per_batch
            )
            assert result is True

        # Verify sync info
        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info

        assert sync_info.received_batches == set(range(batch_count))
        assert sync_info.received_keys_count == expected_keys_count

    def test_receive_batch_wrong_sync_id(self):
        """Test batch receipt with wrong sync ID is rejected."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_correct", 100, 5)

        result = tracker.receive_batch("instance_1", 0, "wrong_sync_id", 0, 20)

        assert result is False

        # Verify no keys were added
        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert len(sync_info.received_batches) == 0
        assert sync_info.received_keys_count == 0

    def test_complete_sync_success(self):
        """Test successful sync completion."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        # Receive all batches
        for i in range(5):
            tracker.receive_batch("instance_1", 0, "sync_1", i, 20)

        result = tracker.complete_sync("instance_1", 0, "sync_1", 100)

        assert result is True
        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.COMPLETED

    def test_complete_sync_wrong_id(self):
        """Test sync completion with wrong sync ID fails."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_correct", 100, 5)

        result = tracker.complete_sync("instance_1", 0, "sync_wrong", 100)

        assert result is False

    def test_mark_failed(self):
        """Test marking a sync as failed."""
        tracker, registry_tree = TestHelper.create_tracker()
        TestHelper.register_worker(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_1", 100, 5)

        tracker.mark_failed("instance_1", 0, "timeout_reason")

        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.FAILED

    def test_check_sync_timeout(self):
        """Test sync timeout detection."""
        tracker, registry_tree = TestHelper.create_tracker(sync_timeout_s=0.1)
        tracker.set_need_full_sync_all(False)  # Disable global flag to test retry logic
        TestHelper.register_worker(registry_tree, "instance_1", 0)
        tracker.start_sync("instance_1", 0, "sync_timeout_test", 100, 5)

        # Wait for timeout
        time.sleep(0.2)

        tracker.check_sync_timeout()

        worker_node = registry_tree.get_worker("instance_1", 0)
        sync_info = worker_node.sync_info
        assert sync_info.state == FullSyncState.FAILED

        # Should need re-sync
        need_sync, reason = tracker.should_request_full_sync("instance_1", 0)
        assert need_sync is True
        assert reason == "sync_failed_retry"

    @pytest.mark.parametrize(
        "total_workers,completed_workers,expected_progress",
        [
            (4, 0, 0.0),  # No workers completed
            (4, 1, 0.25),  # 25%
            (4, 2, 0.5),  # 50%
            (4, 3, 0.75),  # 75%
            (4, 4, 1.0),  # 100%
        ],
    )
    def test_get_global_progress(
        self, total_workers, completed_workers, expected_progress
    ):
        """Test global progress calculation."""
        tracker, registry_tree = TestHelper.create_tracker()
        tracker.set_need_full_sync_all(False)  # Disable global flag

        # Register workers
        for i in range(total_workers):
            TestHelper.register_worker(registry_tree, "instance_1", i)

        # Complete sync for some workers
        for i in range(completed_workers):
            TestHelper.start_and_complete_sync(tracker, "instance_1", i, f"sync_{i}")

        progress = tracker.get_global_progress()
        assert abs(progress - expected_progress) < 0.001

    @pytest.mark.parametrize(
        "total_workers,completed_workers,threshold,can_exit",
        [
            (4, 2, 0.5, True),  # 50% >= 50%
            (4, 1, 0.5, False),  # 25% < 50%
            (4, 3, 0.8, False),  # 75% < 80%
            (4, 4, 0.8, True),  # 100% >= 80%
            (4, 4, 0.0, True),  # 100% >= 0% (always true)
            (4, 0, 0.0, True),  # 0% >= 0% (always true with 0 threshold)
        ],
    )
    def test_can_exit_freeze(
        self, total_workers, completed_workers, threshold, can_exit
    ):
        """Test freeze mode exit check."""
        tracker, registry_tree = TestHelper.create_tracker(
            completion_threshold=threshold
        )
        tracker.set_need_full_sync_all(True)

        # Register workers
        for i in range(total_workers):
            TestHelper.register_worker(registry_tree, "instance_1", i)

        # Complete sync for some workers
        for i in range(completed_workers):
            TestHelper.start_and_complete_sync(tracker, "instance_1", i, f"sync_{i}")

        result = tracker.can_exit_freeze()
        assert result == can_exit

        # If can exit, global flag should be disabled
        if can_exit:
            assert tracker._need_full_sync_all is False
        else:
            assert tracker._need_full_sync_all is True

    @pytest.mark.parametrize(
        "sync_state,total_workers,completed_workers,expected_complete,expected_progress,expected_can_exit_freeze",
        [
            ("in_progress", 2, 0, False, 0.0, False),
            ("completed", 2, 1, True, 0.5, True),
            ("completed", 4, 3, True, 0.75, True),
        ],
    )
    def test_get_sync_status(
        self,
        sync_state,
        total_workers,
        completed_workers,
        expected_complete,
        expected_progress,
        expected_can_exit_freeze,
    ):
        """Test getting sync status for a specific worker."""
        tracker, registry_tree = TestHelper.create_tracker(completion_threshold=0.5)
        tracker.set_need_full_sync_all(False)

        # Register workers
        for i in range(total_workers):
            TestHelper.register_worker(registry_tree, "instance_1", i)

        # Setup sync state for first worker
        instance_id, worker_id = "instance_1", 0
        sync_id = "sync_test"

        if sync_state == "completed":
            tracker.start_sync(instance_id, worker_id, sync_id, 100, 5)
            tracker.complete_sync(instance_id, worker_id, sync_id, 100)
        elif sync_state == "in_progress":
            tracker.start_sync(instance_id, worker_id, sync_id, 100, 5)

        # Complete other workers if specified
        for i in range(completed_workers):
            if i != 0:  # Don't double-complete first worker
                TestHelper.start_and_complete_sync(
                    tracker, "instance_1", i, f"sync_{i}"
                )

        # Query sync status
        is_complete, progress, can_exit = tracker.get_sync_status(
            instance_id, worker_id, sync_id
        )

        assert is_complete == expected_complete
        assert abs(progress - expected_progress) < 0.001
        assert can_exit == expected_can_exit_freeze

    @pytest.mark.parametrize(
        "completed_workers,total_workers,expected_count",
        [
            (0, 4, 0),
            (1, 4, 1),
            (3, 4, 3),
            (4, 4, 4),
        ],
    )
    def test_get_completed_count(
        self, completed_workers, total_workers, expected_count
    ):
        """Test getting completed worker count."""
        tracker, registry_tree = TestHelper.create_tracker()
        tracker.set_need_full_sync_all(False)

        # Register workers
        for i in range(total_workers):
            TestHelper.register_worker(registry_tree, "instance_1", i)

        # Complete sync for some workers
        for i in range(completed_workers):
            TestHelper.start_and_complete_sync(tracker, "instance_1", i, f"sync_{i}")

        completed_count = tracker.get_completed_count()
        assert completed_count == expected_count

    @pytest.mark.parametrize(
        "syncing_workers,expected_count",
        [
            (0, 0),
            (2, 2),
            (4, 4),
        ],
    )
    def test_get_syncing_count(self, syncing_workers, expected_count):
        """Test getting syncing worker count."""
        tracker, registry_tree = TestHelper.create_tracker()
        tracker.set_need_full_sync_all(False)

        # Register 4 workers
        for i in range(4):
            TestHelper.register_worker(registry_tree, "instance_1", i)

        # Start sync for some workers
        for i in range(syncing_workers):
            tracker.start_sync("instance_1", i, f"sync_{i}", 100, 5)

        syncing_count = tracker.get_syncing_count()
        assert syncing_count == expected_count

        # Complete one and check count updates
        if syncing_workers > 0:
            tracker.complete_sync("instance_1", 0, "sync_0", 100)
            updated_count = tracker.get_syncing_count()
            assert updated_count == max(0, syncing_workers - 1)
