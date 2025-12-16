# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVController full sync handling."""

# Standard
from unittest.mock import MagicMock
import asyncio
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStatusMsg,
    KVAdmitMsg,
    KVEvictMsg,
)
from lmcache.v1.cache_controller.utils import FullSyncState, RegistryTree


class TestHelper:
    """Helper methods for KVController tests."""

    @staticmethod
    def create_test_controller(
        completion_threshold=0.8, sync_timeout=300.0
    ) -> KVController:
        """Create KVController with test settings."""
        registry = RegistryTree()
        controller = KVController(
            registry=registry,
            full_sync_completion_threshold=completion_threshold,
            full_sync_timeout_s=sync_timeout,
        )
        controller.cluster_executor = MagicMock()
        return controller

    @staticmethod
    def register_worker(controller, instance_id, worker_id, ip="127.0.0.1", port=8000):
        """Register a worker in the registry."""
        controller.registry.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=None,
            socket=MagicMock(),
            registration_time=time.time(),
        )

    @staticmethod
    def start_sync_msg(instance_id, worker_id, sync_id, total_keys=100, batch_count=5):
        """Create a FullSyncStartMsg."""
        return FullSyncStartMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location="LocalCPUBackend",
            sync_id=sync_id,
            total_keys=total_keys,
            batch_count=batch_count,
        )

    @staticmethod
    def batch_msg(instance_id, worker_id, sync_id, batch_id, keys):
        """Create a FullSyncBatchMsg."""
        return FullSyncBatchMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location="LocalCPUBackend",
            sync_id=sync_id,
            batch_id=batch_id,
            keys=keys,
        )

    @staticmethod
    def end_sync_msg(instance_id, worker_id, sync_id, actual_total_keys):
        """Create a FullSyncEndMsg."""
        return FullSyncEndMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location="LocalCPUBackend",
            sync_id=sync_id,
            actual_total_keys=actual_total_keys,
        )


@pytest.fixture
def kv_controller() -> KVController:
    """Create KVController instance for testing."""
    return TestHelper.create_test_controller()


class TestKVControllerFullSyncStart:
    """Test cases for handle_full_sync_start."""

    @pytest.mark.asyncio
    async def test_full_sync_start_clears_existing_keys(self, kv_controller):
        """Test full sync start acceptance and key clearing."""
        instance_id, worker_id, sync_id, location = (
            "instance_1",
            0,
            "sync_123",
            "LocalCPUBackend",
        )

        # Register worker and pre-populate keys
        TestHelper.register_worker(kv_controller, instance_id, worker_id)
        for key in [1, 2, 3, 4, 5]:
            kv_controller.registry.admit_kv(instance_id, worker_id, location, key)

        # Start sync - should be accepted and clear keys
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 100, 5)
        ret_msg = await kv_controller.handle_full_sync_start(start_msg)

        assert ret_msg.accepted is True
        assert kv_controller.full_sync_tracker.is_worker_syncing(instance_id, worker_id)

        # Verify keys cleared
        keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert len(keys) == 0

    @pytest.mark.asyncio
    async def test_sync_start_conflict(self, kv_controller):
        """Test handling of sync start conflict."""
        instance_id, worker_id = "instance_1", 0
        TestHelper.register_worker(kv_controller, instance_id, worker_id)

        # Start first sync
        start_msg1 = TestHelper.start_sync_msg(
            instance_id, worker_id, "sync_123", 1000, 10
        )
        ret1 = await kv_controller.handle_full_sync_start(start_msg1)
        assert ret1.accepted is True

        # Try to start with different ID - should reject
        start_msg2 = TestHelper.start_sync_msg(
            instance_id, worker_id, "sync_456", 2000, 20
        )
        ret2 = await kv_controller.handle_full_sync_start(start_msg2)
        assert ret2.accepted is False
        assert ret2.error_msg is not None

        # Try to start with same ID - should accept (retry)
        start_msg3 = TestHelper.start_sync_msg(
            instance_id, worker_id, "sync_123", 1000, 10
        )
        ret3 = await kv_controller.handle_full_sync_start(start_msg3)
        assert ret3.accepted is True


class TestKVControllerFullSyncBatch:
    """Test cases for handle_full_sync_batch."""

    @pytest.mark.asyncio
    async def test_batch_adds_keys(self, kv_controller):
        """Test that batch adds keys to registry."""
        instance_id, worker_id, sync_id, location = (
            "instance_1",
            0,
            "sync_123",
            "LocalCPUBackend",
        )

        # Register and start sync
        TestHelper.register_worker(kv_controller, instance_id, worker_id)
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 100, 5)
        await kv_controller.handle_full_sync_start(start_msg)

        # Send batch
        keys = [1, 2, 3, 4, 5]
        batch_msg = TestHelper.batch_msg(instance_id, worker_id, sync_id, 0, keys)
        await kv_controller.handle_full_sync_batch(batch_msg)

        # Verify keys added
        actual_keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert len(actual_keys) == 5
        for key in keys:
            assert key in actual_keys

    @pytest.mark.asyncio
    async def test_multiple_batches(self, kv_controller):
        """Test multiple batch messages."""
        instance_id, worker_id, sync_id, location = (
            "instance_1",
            0,
            "sync_123",
            "LocalCPUBackend",
        )

        # Register and start sync
        TestHelper.register_worker(kv_controller, instance_id, worker_id)
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 15, 3)
        await kv_controller.handle_full_sync_start(start_msg)

        # Send 3 batches
        for batch_id in range(3):
            keys = list(range(batch_id * 5, (batch_id + 1) * 5))
            batch_msg = TestHelper.batch_msg(
                instance_id, worker_id, sync_id, batch_id, keys
            )
            await kv_controller.handle_full_sync_batch(batch_msg)

        # Verify all keys added
        actual_keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert len(actual_keys) == 15
        for i in range(15):
            assert i in actual_keys

    @pytest.mark.asyncio
    async def test_batch_with_wrong_sync_id_rejected(self, kv_controller):
        """Test that batch with wrong sync ID is rejected."""
        instance_id, worker_id, location = "instance_1", 0, "LocalCPUBackend"

        # Register and start sync with correct ID
        TestHelper.register_worker(kv_controller, instance_id, worker_id)
        start_msg = TestHelper.start_sync_msg(
            instance_id, worker_id, "sync_correct", 10, 1
        )
        await kv_controller.handle_full_sync_start(start_msg)

        # Record keys before batch
        keys_before = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        ).copy()

        # Send batch with wrong sync ID
        batch_msg = TestHelper.batch_msg(
            instance_id, worker_id, "sync_wrong", 0, [1, 2, 3, 4, 5]
        )
        await kv_controller.handle_full_sync_batch(batch_msg)

        # Keys should NOT be added
        keys_after = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert keys_after == keys_before


class TestKVControllerFullSyncEnd:
    """Test cases for handle_full_sync_end."""

    @pytest.mark.asyncio
    async def test_sync_end_completes_sync(self, kv_controller):
        """Test that sync end marks sync as completed."""
        instance_id, worker_id, sync_id = "instance_1", 0, "sync_123"

        # Register and start sync
        TestHelper.register_worker(kv_controller, instance_id, worker_id)
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 5, 1)
        await kv_controller.handle_full_sync_start(start_msg)

        # Send batch
        batch_msg = TestHelper.batch_msg(
            instance_id, worker_id, sync_id, 0, [1, 2, 3, 4, 5]
        )
        await kv_controller.handle_full_sync_batch(batch_msg)

        # End sync
        end_msg = TestHelper.end_sync_msg(instance_id, worker_id, sync_id, 5)
        await kv_controller.handle_full_sync_end(end_msg)

        # Verify sync is completed
        assert not kv_controller.full_sync_tracker.is_worker_syncing(
            instance_id, worker_id
        )
        worker_node = kv_controller.registry.get_worker(instance_id, worker_id)
        assert worker_node.sync_info.state == FullSyncState.COMPLETED


class TestKVControllerFullSyncStatus:
    """Test cases for handle_full_sync_status."""

    @pytest.mark.asyncio
    async def test_sync_status_with_progress(self, kv_controller):
        """Test status query with progress calculation."""
        # Set a lower threshold for easier testing
        kv_controller.full_sync_tracker.completion_threshold = 0.5

        # Register 4 workers
        for i in range(4):
            TestHelper.register_worker(kv_controller, "instance_1", i)

        # Complete sync for 2 workers (50%)
        for i in range(2):
            sync_id = f"sync_{i}"
            kv_controller.full_sync_tracker.start_sync("instance_1", i, sync_id, 100, 5)
            kv_controller.full_sync_tracker.complete_sync("instance_1", i, sync_id, 100)

        # Query status for first worker
        status_msg = FullSyncStatusMsg(
            instance_id="instance_1", worker_id=0, sync_id="sync_0"
        )
        status_ret = await kv_controller.handle_full_sync_status(status_msg)

        assert status_ret.is_complete is True
        assert abs(status_ret.global_progress - 0.5) < 0.001
        assert status_ret.can_exit_freeze is True  # 50% >= 50% threshold

    @pytest.mark.asyncio
    async def test_sync_status_incomplete(self, kv_controller):
        """Test status query for incomplete sync."""
        instance_id, worker_id, sync_id = "instance_1", 0, "sync_123"

        # Register 2 workers
        for i in range(2):
            TestHelper.register_worker(kv_controller, instance_id, i)

        # Start sync but don't complete
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 100, 5)
        await kv_controller.handle_full_sync_start(start_msg)

        # Query status
        status_msg = FullSyncStatusMsg(
            instance_id=instance_id, worker_id=worker_id, sync_id=sync_id
        )
        status_ret = await kv_controller.handle_full_sync_status(status_msg)

        assert status_ret.is_complete is False
        assert status_ret.global_progress == 0.0
        assert status_ret.can_exit_freeze is False


class TestKVControllerIncrementalDiscardDuringSync:
    """Test cases for incremental message discard during sync."""

    @pytest.mark.asyncio
    async def test_incremental_messages_discarded_during_sync(self, kv_controller):
        """Test that incremental messages are discarded during sync."""
        instance_id, worker_id, location = "instance_1", 0, "LocalCPUBackend"
        sync_id = "sync_test"

        # Register worker
        TestHelper.register_worker(kv_controller, instance_id, worker_id)

        # Start sync
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 5, 1)
        await kv_controller.handle_full_sync_start(start_msg)

        # Try admit - should be discarded
        admit_msg = KVAdmitMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=888,
            location=location,
            seq_num=0,
        )
        await kv_controller.admit(admit_msg)
        keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert 888 not in keys

        # Add a key via batch, then try to evict it - evict should be discarded
        batch_msg = TestHelper.batch_msg(instance_id, worker_id, sync_id, 0, [999])
        await kv_controller.handle_full_sync_batch(batch_msg)

        evict_msg = KVEvictMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=999,
            location=location,
            seq_num=0,
        )
        await kv_controller.evict(evict_msg)
        keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert 999 in keys  # Key should still be there (evict discarded)

    @pytest.mark.asyncio
    async def test_incremental_allowed_after_sync_complete(self, kv_controller):
        """Test that incremental messages work after sync is complete."""
        instance_id, worker_id, location = "instance_1", 0, "LocalCPUBackend"
        sync_id = "sync_123"

        # Register worker and complete sync
        TestHelper.register_worker(kv_controller, instance_id, worker_id)
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 0, 1)
        await kv_controller.handle_full_sync_start(start_msg)
        end_msg = TestHelper.end_sync_msg(instance_id, worker_id, sync_id, 0)
        await kv_controller.handle_full_sync_end(end_msg)

        # Now incremental admit should work
        admit_msg = KVAdmitMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=1000,
            location=location,
            seq_num=1,
        )
        await kv_controller.admit(admit_msg)

        # Verify key was added
        keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert 1000 in keys


class TestKVControllerSyncTimeout:
    """Test cases for sync timeout handling."""

    @pytest.mark.asyncio
    async def test_sync_timeout_marks_failed(self):
        """Test that sync timeout marks worker as failed."""
        # Create controller with short timeout for testing
        controller = TestHelper.create_test_controller(
            sync_timeout=0.1
        )  # 100ms timeout
        controller.full_sync_tracker.sync_timeout_s = 0.1

        # Disable global flag to test retry logic
        controller.full_sync_tracker.set_need_full_sync_all(False)

        # Register worker
        instance_id, worker_id = "instance_1", 0
        controller.registry.register_worker(
            instance_id=instance_id,
            worker_id=worker_id,
            ip="127.0.0.1",
            port=8000,
            peer_init_url=None,
            socket=MagicMock(),
            registration_time=time.time(),
        )

        # Start sync but don't complete it
        start_msg = TestHelper.start_sync_msg(
            instance_id, worker_id, "sync_timeout_test", 100, 5
        )
        await controller.handle_full_sync_start(start_msg)

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Check timeout
        controller.full_sync_tracker.check_sync_timeout()

        # Worker should be marked as failed
        worker_node = controller.registry.get_worker(instance_id, worker_id)
        assert worker_node.sync_info.state == FullSyncState.FAILED

        # Should need re-sync
        need_sync, reason = controller.full_sync_tracker.should_request_full_sync(
            instance_id, worker_id
        )
        assert need_sync is True
        assert reason == "sync_failed_retry"
