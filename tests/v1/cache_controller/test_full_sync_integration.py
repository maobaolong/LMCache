# SPDX-License-Identifier: Apache-2.0
"""Integration tests for full sync functionality."""

# Standard
from unittest.mock import MagicMock, patch
import asyncio
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.commands.full_sync import FullSyncCommand
from lmcache.v1.cache_controller.controllers.full_sync_tracker import FullSyncTracker
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.controllers.registration_controller import (
    RegistrationController,
)
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStatusMsg,
    HeartbeatMsg,
    KVAdmitMsg,
    Msg,
    RegisterMsg,
)
from lmcache.v1.cache_controller.utils import FullSyncState, RegistryTree


def create_test_key(key_id: int) -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, key_id, torch.bfloat16)


class MockZMQSocket:
    """Mock ZMQ socket for testing."""

    def __init__(self):
        self.sent_messages = []

    def send(self, data):
        self.sent_messages.append(data)


class TestHelper:
    """Helper class with common test utilities."""

    @staticmethod
    async def register_worker(
        controller, instance_id, worker_id, ip="192.168.1.1", port=8000
    ):
        """Helper to register a worker."""
        with patch(
            "lmcache.v1.cache_controller.controllers.registration_controller.get_zmq_socket"
        ) as mock_socket:
            mock_socket.return_value = MockZMQSocket()
            register_msg = RegisterMsg(
                instance_id=instance_id,
                worker_id=worker_id,
                ip=ip,
                port=port,
                peer_init_url=None,
            )
            return await controller.register(register_msg)

    @staticmethod
    async def send_heartbeat(
        controller, instance_id, worker_id, ip="192.168.1.1", port=8000
    ):
        """Helper to send heartbeat."""
        heartbeat_msg = HeartbeatMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=None,
        )
        return await controller.heartbeat(heartbeat_msg)

    @staticmethod
    def start_sync_msg(instance_id, worker_id, sync_id, total_keys=10, batch_count=2):
        """Helper to create FullSyncStartMsg."""
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
        """Helper to create FullSyncBatchMsg."""
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
        """Helper to create FullSyncEndMsg."""
        return FullSyncEndMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location="LocalCPUBackend",
            sync_id=sync_id,
            actual_total_keys=actual_total_keys,
        )


@pytest.fixture
def shared_registry():
    return RegistryTree()


@pytest.fixture
def kv_controller(shared_registry):
    controller = KVController(
        registry=shared_registry,
        full_sync_completion_threshold=0.5,
        full_sync_timeout_s=300.0,
    )
    controller.cluster_executor = MagicMock()
    return controller


@pytest.fixture
def registration_controller(kv_controller, shared_registry):
    controller = RegistrationController()
    controller.registry = shared_registry
    controller.kv_controller = kv_controller
    controller.cluster_executor = MagicMock()
    return controller


class TestFullSyncIntegrationFlow:
    """Integration tests for complete full sync flow."""

    @pytest.mark.asyncio
    async def test_complete_sync_flow_single_worker(
        self, kv_controller, registration_controller
    ):
        """Test complete sync flow for single worker."""
        instance_id = "test_instance"
        worker_id = 0
        sync_id = "sync_test"
        location = "LocalCPUBackend"

        # Register worker
        await TestHelper.register_worker(
            registration_controller, instance_id, worker_id
        )

        # Pre-populate some keys that should be cleared during sync
        for key in [100, 200, 300]:
            kv_controller.registry.admit_kv(instance_id, worker_id, location, key)

        # Check heartbeat returns FullSyncCommand
        heartbeat_ret = await TestHelper.send_heartbeat(
            registration_controller, instance_id, worker_id
        )
        assert len(heartbeat_ret.commands) == 1
        assert isinstance(heartbeat_ret.commands[0], FullSyncCommand)
        assert heartbeat_ret.commands[0].reason == "controller_restart"

        # Start sync
        start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id)
        start_ret = await kv_controller.handle_full_sync_start(start_msg)

        assert start_ret.accepted is True
        assert kv_controller.full_sync_tracker.is_worker_syncing(instance_id, worker_id)

        # Old keys should be cleared
        assert (
            len(
                kv_controller.registry.get_worker_kv_keys(
                    instance_id, worker_id, location
                )
            )
            == 0
        )

        # Send batches
        await kv_controller.handle_full_sync_batch(
            TestHelper.batch_msg(instance_id, worker_id, sync_id, 0, [1, 2, 3, 4, 5])
        )
        await kv_controller.handle_full_sync_batch(
            TestHelper.batch_msg(instance_id, worker_id, sync_id, 1, [6, 7, 8, 9, 10])
        )

        # Verify keys
        keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert len(keys) == 10
        assert keys == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

        # End sync
        end_msg = TestHelper.end_sync_msg(instance_id, worker_id, sync_id, 10)
        await kv_controller.handle_full_sync_end(end_msg)

        # Verify no more sync needed
        heartbeat_ret2 = await TestHelper.send_heartbeat(
            registration_controller, instance_id, worker_id
        )
        assert len(heartbeat_ret2.commands) == 0

        # Verify sync is completed
        assert not kv_controller.full_sync_tracker.is_worker_syncing(
            instance_id, worker_id
        )
        worker_node = kv_controller.registry.get_worker(instance_id, worker_id)
        assert worker_node.sync_info.state == FullSyncState.COMPLETED

        # Query status
        status_msg = FullSyncStatusMsg(
            instance_id=instance_id, worker_id=worker_id, sync_id=sync_id
        )
        status_ret = await kv_controller.handle_full_sync_status(status_msg)

        assert status_ret.is_complete is True
        assert status_ret.global_progress == 1.0
        assert status_ret.can_exit_freeze is True

    @pytest.mark.asyncio
    async def test_sync_progress_multiple_workers(
        self, kv_controller, registration_controller
    ):
        """Test sync progress calculation with multiple workers."""
        workers = [("instance_1", i, f"192.168.1.{i + 1}") for i in range(4)]

        # Register all workers
        for instance_id, worker_id, ip in workers:
            await TestHelper.register_worker(
                registration_controller, instance_id, worker_id, ip
            )

        # Complete sync for 2 workers (50%)
        for i in range(2):
            instance_id, worker_id, _ = workers[i]
            sync_id = f"sync_{instance_id}_{worker_id}"

            # Start sync
            start_msg = TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 5, 1)
            await kv_controller.handle_full_sync_start(start_msg)

            # Send batch
            batch_msg = TestHelper.batch_msg(
                instance_id, worker_id, sync_id, 0, list(range(i * 5, (i + 1) * 5))
            )
            await kv_controller.handle_full_sync_batch(batch_msg)

            # End sync
            end_msg = TestHelper.end_sync_msg(instance_id, worker_id, sync_id, 5)
            await kv_controller.handle_full_sync_end(end_msg)

        # Check progress (50% with default 0.5 threshold)
        progress = kv_controller.full_sync_tracker.get_global_progress()
        assert progress == 0.5

        # Check freeze exit
        can_exit = kv_controller.full_sync_tracker.can_exit_freeze()
        assert can_exit is True

    @pytest.mark.asyncio
    async def test_incremental_messages_discarded_during_sync(
        self, kv_controller, registration_controller
    ):
        """Test incremental messages are discarded during sync."""
        instance_id = "test_instance"
        worker_id = 0
        sync_id = "sync_test"
        location = "LocalCPUBackend"

        # Register and start sync
        await TestHelper.register_worker(
            registration_controller, instance_id, worker_id
        )
        await kv_controller.handle_full_sync_start(
            TestHelper.start_sync_msg(instance_id, worker_id, sync_id, 5, 1)
        )

        # Try incremental admit - should be discarded
        admit_msg = KVAdmitMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=999,
            location=location,
            seq_num=0,
        )
        await kv_controller.admit(admit_msg)

        keys = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert 999 not in keys

        # Complete sync
        await kv_controller.handle_full_sync_batch(
            TestHelper.batch_msg(instance_id, worker_id, sync_id, 0, [1, 2, 3, 4, 5])
        )
        await kv_controller.handle_full_sync_end(
            TestHelper.end_sync_msg(instance_id, worker_id, sync_id, 5)
        )

        # Now incremental admit should work
        admit_msg2 = KVAdmitMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            key=1000,
            location=location,
            seq_num=1,
        )
        await kv_controller.admit(admit_msg2)

        keys_final = kv_controller.registry.get_worker_kv_keys(
            instance_id, worker_id, location
        )
        assert 1000 in keys_final


class TestFullSyncErrorHandling:
    """Tests for error handling in full sync."""

    @pytest.mark.asyncio
    async def test_sync_timeout_marks_failed(self, kv_controller, shared_registry):
        """Test sync timeout marks worker as failed."""
        # Use short timeout for testing
        kv_controller.full_sync_tracker = FullSyncTracker(
            registry_tree=kv_controller.registry,
            completion_threshold=0.8,
            sync_timeout_s=0.1,  # 100ms
        )
        kv_controller.full_sync_tracker.set_need_full_sync_all(False)

        # Register worker
        shared_registry.register_worker(
            instance_id="instance_1",
            worker_id=0,
            ip="192.168.1.1",
            port=8000,
            peer_init_url=None,
            socket=None,
            registration_time=time.time(),
        )

        # Start sync but don't complete it
        await kv_controller.handle_full_sync_start(
            TestHelper.start_sync_msg("instance_1", 0, "sync_timeout_test", 100, 5)
        )

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Check timeout
        kv_controller.full_sync_tracker.check_sync_timeout()

        # Worker should be marked as failed
        worker_node = kv_controller.registry.get_worker("instance_1", 0)
        assert worker_node.sync_info.state == FullSyncState.FAILED

        # Should need re-sync
        need_sync, reason = kv_controller.full_sync_tracker.should_request_full_sync(
            "instance_1", 0
        )
        assert need_sync is True
        assert reason == "sync_failed_retry"


class TestMessageSerialization:
    """Tests for message serialization."""

    def test_large_batch_serialization(self):
        """Test serialization of large batch messages."""
        keys = list(range(100000))  # 100K keys
        msg = TestHelper.batch_msg("test_instance", 0, "sync_large", 0, keys)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncBatchMsg)
        assert len(decoded.keys) == 100000
        assert decoded.keys == keys
