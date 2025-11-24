# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import ctypes
import time

# Third Party
import pytest

# First Party
from lmcache.v1.transfer_channel import CreateTransferChannel


class TestSocketChannel:
    """Test suite for SocketChannel implementation"""

    def test_create_socket_channel(self):
        """Test creating a socket channel"""
        buffer_size = 1024 * 1024
        buffer = (ctypes.c_byte * buffer_size)()
        buffer_ptr = ctypes.addressof(buffer)

        channel = CreateTransferChannel(
            channel_type="socket",
            async_mode=False,
            role="both",
            buffer_ptr=buffer_ptr,
            buffer_size=buffer_size,
            align_bytes=4096,
            tp_rank=0,
            peer_init_url="127.0.0.1:5555",
            data_port=6666,
        )

        assert channel is not None
        channel.close()

    def test_sync_send_recv(self):
        """Test synchronous send and receive"""
        buffer_size = 1024 * 1024
        sender_buffer = (ctypes.c_byte * buffer_size)()
        receiver_buffer = (ctypes.c_byte * buffer_size)()

        sender = CreateTransferChannel(
            channel_type="socket",
            async_mode=False,
            role="sender",
            buffer_ptr=ctypes.addressof(sender_buffer),
            buffer_size=buffer_size,
            align_bytes=4096,
            tp_rank=0,
            peer_init_url="127.0.0.1:5556",
            data_port=6667,
        )

        receiver = CreateTransferChannel(
            channel_type="socket",
            async_mode=False,
            role="receiver",
            buffer_ptr=ctypes.addressof(receiver_buffer),
            buffer_size=buffer_size,
            align_bytes=4096,
            tp_rank=0,
            peer_init_url="127.0.0.1:5557",
            data_port=6668,
        )

        time.sleep(0.1)

        test_data = [b"Hello", b"World", b"Test"]
        recv_buffers = [bytearray(len(d)) for d in test_data]

        def send_task():
            sender.batched_send(test_data, transfer_spec={"peer_url": "127.0.0.1:6668"})

        def recv_task():
            receiver.batched_recv(
                recv_buffers, transfer_spec={"peer_url": "127.0.0.1:6667"}
            )

        # Standard
        import threading

        send_thread = threading.Thread(target=send_task)
        recv_thread = threading.Thread(target=recv_task)

        recv_thread.start()
        time.sleep(0.1)
        send_thread.start()

        send_thread.join()
        recv_thread.join()

        for i, data in enumerate(test_data):
            assert bytes(recv_buffers[i]) == data

        sender.close()
        receiver.close()

    @pytest.mark.asyncio
    async def test_async_send_recv(self):
        """Test asynchronous send and receive"""
        buffer_size = 1024 * 1024
        sender_buffer = (ctypes.c_byte * buffer_size)()
        receiver_buffer = (ctypes.c_byte * buffer_size)()

        loop = asyncio.get_event_loop()

        sender = CreateTransferChannel(
            channel_type="socket",
            async_mode=True,
            role="sender",
            buffer_ptr=ctypes.addressof(sender_buffer),
            buffer_size=buffer_size,
            align_bytes=4096,
            tp_rank=0,
            peer_init_url="127.0.0.1:5558",
            data_port=6669,
            event_loop=loop,
        )

        receiver = CreateTransferChannel(
            channel_type="socket",
            async_mode=True,
            role="receiver",
            buffer_ptr=ctypes.addressof(receiver_buffer),
            buffer_size=buffer_size,
            align_bytes=4096,
            tp_rank=0,
            peer_init_url="127.0.0.1:5559",
            data_port=6670,
            event_loop=loop,
        )

        await asyncio.sleep(0.1)

        test_data = [b"Async", b"Test", b"Data"]
        recv_buffers = [bytearray(len(d)) for d in test_data]

        async def send_task():
            await sender.async_batched_send(
                test_data, transfer_spec={"peer_url": "127.0.0.1:6670"}
            )

        async def recv_task():
            await receiver.async_batched_recv(
                recv_buffers, transfer_spec={"peer_url": "127.0.0.1:6669"}
            )

        await asyncio.gather(recv_task(), send_task())

        for i, data in enumerate(test_data):
            assert bytes(recv_buffers[i]) == data

        sender.close()
        receiver.close()

    def test_channel_registry(self):
        """Test that both nixl and socket channels are registered"""
        # First Party
        from lmcache.v1.transfer_channel import CHANNEL_REGISTRY

        assert "nixl" in CHANNEL_REGISTRY
        assert "socket" in CHANNEL_REGISTRY

    def test_unsupported_channel_type(self):
        """Test that unsupported channel type raises error"""
        buffer_size = 1024 * 1024
        buffer = (ctypes.c_byte * buffer_size)()

        with pytest.raises(ValueError, match="Unsupported channel type"):
            CreateTransferChannel(
                channel_type="unsupported",
                async_mode=False,
                role="both",
                buffer_ptr=ctypes.addressof(buffer),
                buffer_size=buffer_size,
                align_bytes=4096,
                tp_rank=0,
                peer_init_url="127.0.0.1:5560",
            )
