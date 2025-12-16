# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Full Sync message types."""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.cache_controller.message import (
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStartMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
    HeartbeatMsg,
    Msg,
)


class TestHeartbeatMessages:
    """Test cases for Heartbeat message types."""

    @pytest.mark.parametrize(
        "instance_id,worker_id,ip,port,peer_init_url",
        [
            ("test_instance", 1, "192.168.1.2", 8001, None),
            ("instance_2", 3, "192.168.2.1", 9000, "tcp://192.168.2.1:9000"),
        ],
    )
    def test_heartbeat_msg_serialization(
        self, instance_id, worker_id, ip, port, peer_init_url
    ):
        """Test HeartbeatMsg creation and serialization."""
        msg = HeartbeatMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            ip=ip,
            port=port,
            peer_init_url=peer_init_url,
        )
        # Test serialization
        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HeartbeatMsg)
        assert decoded.instance_id == instance_id
        assert decoded.worker_id == worker_id
        assert decoded.peer_init_url == peer_init_url


class TestFullSyncStartMessages:
    """Test cases for FullSyncStart message types."""

    @pytest.mark.parametrize(
        "instance_id,worker_id,sync_id,total_keys,batch_count",
        [
            ("instance_1", 2, "sync_abc", 5000, 25),
        ],
    )
    def test_full_sync_start_msg_serialization(
        self, instance_id, worker_id, sync_id, total_keys, batch_count
    ):
        """Test FullSyncStartMsg creation and serialization."""
        msg = FullSyncStartMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location="LocalCPUBackend",
            sync_id=sync_id,
            total_keys=total_keys,
            batch_count=batch_count,
        )
        # Test serialization
        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStartMsg)
        assert decoded.sync_id == sync_id
        assert decoded.total_keys == total_keys


class TestFullSyncBatchMessages:
    """Test cases for FullSyncBatch message type."""

    @pytest.mark.parametrize(
        "instance_id,worker_id,sync_id,batch_id,keys",
        [
            ("instance_1", 1, "sync_batch_test", 3, [100, 200, 300, 400, 500]),
            ("test_instance", 0, "sync_12345", 5, list(range(2000))),
        ],
    )
    def test_full_sync_batch_msg_serialization(
        self, instance_id, worker_id, sync_id, batch_id, keys
    ):
        """Test FullSyncBatchMsg creation and serialization."""
        msg = FullSyncBatchMsg(
            instance_id=instance_id,
            worker_id=worker_id,
            location="LocalCPUBackend",
            sync_id=sync_id,
            batch_id=batch_id,
            keys=keys,
        )
        # Test serialization
        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncBatchMsg)
        assert decoded.batch_id == batch_id
        assert decoded.keys == keys


class TestFullSyncEndMessages:
    """Test cases for FullSyncEnd message type."""

    def test_full_sync_end_msg(self):
        """Test FullSyncEndMsg creation and serialization."""
        msg = FullSyncEndMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_12345",
            actual_total_keys=1000,
        )
        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncEndMsg)
        assert decoded.actual_total_keys == 1000


class TestFullSyncStatusMessages:
    """Test cases for FullSyncStatus message types."""

    def test_full_sync_status_msg(self):
        """Test FullSyncStatusMsg creation and serialization."""
        msg = FullSyncStatusMsg(
            instance_id="test_instance",
            worker_id=0,
            sync_id="sync_12345",
        )
        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStatusMsg)
        assert decoded.sync_id == "sync_12345"

    def test_full_sync_status_ret_msg(self):
        """Test FullSyncStatusRetMsg creation and serialization."""
        msg = FullSyncStatusRetMsg(
            sync_id="sync_12345",
            is_complete=True,
            global_progress=0.85,
            can_exit_freeze=True,
        )
        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, FullSyncStatusRetMsg)
        assert decoded.global_progress == 0.85
        assert decoded.can_exit_freeze is True


class TestMessageDescribe:
    """Test cases for message describe() methods."""

    def test_message_describe(self):
        """Test message describe methods for key message types."""
        # Test FullSyncStartMsg
        start_msg = FullSyncStartMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            total_keys=1000,
            batch_count=10,
        )
        assert "sync_123" in start_msg.describe()
        assert "1000" in start_msg.describe()

        # Test FullSyncBatchMsg
        batch_msg = FullSyncBatchMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            sync_id="sync_123",
            batch_id=5,
            keys=[1, 2, 3],
        )
        assert "sync_123" in batch_msg.describe()
        assert "5" in batch_msg.describe()
