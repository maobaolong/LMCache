# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Union
import asyncio
import socket
import struct

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.transfer_channel.py_socket_channel import PySocketChannel

logger = init_logger(__name__)


class DataTransferMsg(msgspec.Struct, tag=True):
    """Message for data transfer metadata"""

    num_objects: int
    sizes: list[int]


class SocketChannel(PySocketChannel):
    """
    Complete socket-based transfer channel with data plane implementation.

    Control plane: Inherited from PySocketChannel (ZMQ-based handshake)
    Data plane: Native Python sockets for data transfer
    """

    def __init__(
        self,
        async_mode: bool = False,
        **kwargs,
    ):
        super().__init__(async_mode=async_mode, **kwargs)

        self.data_port = kwargs.get("data_port", 0)
        self.data_socket: Optional[socket.socket] = None
        self.peer_data_sockets: dict[str, socket.socket] = {}

        self._init_data_socket()

    def _init_data_socket(self):
        """Initialize data socket for receiving data"""
        if self.role in ["both", "receiver"]:
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_socket.bind(("0.0.0.0", self.data_port))
            self.data_socket.listen(5)

            if self.data_port == 0:
                self.data_port = self.data_socket.getsockname()[1]

            logger.info(f"Data socket listening on port {self.data_port}")

    def _on_peer_connected(self, peer_url: str):
        """Setup data connection when a peer connects"""
        if self.role in ["both", "sender"]:
            host, port = self._parse_peer_url(peer_url)
            data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_sock.connect((host, port))
            self.peer_data_sockets[peer_url] = data_sock
            logger.info(f"Connected to peer data socket at {host}:{port}")

    def _parse_peer_url(self, peer_url: str) -> tuple[str, int]:
        """Parse peer URL to extract host and port"""
        if "://" in peer_url:
            peer_url = peer_url.split("://")[1]
        host, port = peer_url.rsplit(":", 1)
        return host, int(port)

    def _send_data(self, sock: socket.socket, data: bytes):
        """Send data with length prefix"""
        length = len(data)
        sock.sendall(struct.pack("!Q", length))
        sock.sendall(data)

    def _recv_data(self, sock: socket.socket) -> bytes:
        """Receive data with length prefix"""
        length_bytes = self._recv_exact(sock, 8)
        length = struct.unpack("!Q", length_bytes)[0]
        return self._recv_exact(sock, length)

    def _recv_exact(self, sock: socket.socket, size: int) -> bytes:
        """Receive exact number of bytes"""
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Socket connection broken")
            data.extend(chunk)
        return bytes(data)

    async def _async_send_data(self, sock: socket.socket, data: bytes):
        """Async send data with length prefix"""
        loop = asyncio.get_event_loop()
        length = len(data)
        await loop.sock_sendall(sock, struct.pack("!Q", length))
        await loop.sock_sendall(sock, data)

    async def _async_recv_data(self, sock: socket.socket) -> bytes:
        """Async receive data with length prefix"""
        length_bytes = await self._async_recv_exact(sock, 8)
        length = struct.unpack("!Q", length_bytes)[0]
        return await self._async_recv_exact(sock, length)

    async def _async_recv_exact(self, sock: socket.socket, size: int) -> bytes:
        """Async receive exact number of bytes"""
        loop = asyncio.get_event_loop()
        data = bytearray()
        while len(data) < size:
            chunk = await loop.sock_recv(sock, size - len(data))
            if not chunk:
                raise ConnectionError("Socket connection broken")
            data.extend(chunk)
        return bytes(data)

    def get_local_mem_indices(
        self, objects: Union[list[bytes], list[MemoryObj]]
    ) -> list[int]:
        """Get memory indices for objects"""
        return list(range(len(objects)))

    def batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Send a batch of data through socket"""
        assert transfer_spec is not None
        peer_url = transfer_spec.get("peer_url")
        assert peer_url in self.peer_data_sockets

        sock = self.peer_data_sockets[peer_url]

        data_list = []
        for obj in objects:
            if isinstance(obj, bytes):
                data_list.append(obj)
            elif isinstance(obj, MemoryObj):
                data_list.append(obj.byte_array)
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        sizes = [len(data) for data in data_list]
        meta_msg = DataTransferMsg(num_objects=len(objects), sizes=sizes)
        meta_bytes = msgspec.msgpack.encode(meta_msg)

        self._send_data(sock, meta_bytes)

        for data in data_list:
            self._send_data(sock, data)

        return len(objects)

    def batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Receive a batch of data through socket"""
        assert transfer_spec is not None
        peer_url = transfer_spec.get("peer_url")
        assert peer_url is not None, "peer_url must be provided in transfer_spec"

        if peer_url not in self.peer_data_sockets:
            assert self.data_socket is not None, "Data socket not initialized"
            conn, _ = self.data_socket.accept()
            self.peer_data_sockets[peer_url] = conn

        sock = self.peer_data_sockets[peer_url]

        meta_bytes = self._recv_data(sock)
        meta_msg = msgspec.msgpack.decode(meta_bytes, type=DataTransferMsg)

        assert meta_msg.num_objects == len(buffers)

        for i, buffer in enumerate(buffers):
            data = self._recv_data(sock)
            if isinstance(buffer, MemoryObj):
                # Standard
                import ctypes

                ctypes.memmove(buffer.data_ptr, data, len(data))
            elif isinstance(buffer, bytearray):
                buffer[:] = data
            else:
                raise ValueError(f"Unsupported buffer type: {type(buffer)}")

        return len(buffers)

    async def async_batched_send(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async send a batch of data through socket"""
        assert transfer_spec is not None
        peer_url = transfer_spec.get("peer_url")
        assert peer_url in self.peer_data_sockets

        sock = self.peer_data_sockets[peer_url]

        data_list = []
        for obj in objects:
            if isinstance(obj, bytes):
                data_list.append(obj)
            elif isinstance(obj, MemoryObj):
                data_list.append(obj.byte_array)
            else:
                raise ValueError(f"Unsupported object type: {type(obj)}")

        sizes = [len(data) for data in data_list]
        meta_msg = DataTransferMsg(num_objects=len(objects), sizes=sizes)
        meta_bytes = msgspec.msgpack.encode(meta_msg)

        await self._async_send_data(sock, meta_bytes)

        for data in data_list:
            await self._async_send_data(sock, data)

        return len(objects)

    async def async_batched_recv(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async receive a batch of data through socket"""
        assert transfer_spec is not None
        peer_url = transfer_spec.get("peer_url")
        assert peer_url is not None, "peer_url must be provided in transfer_spec"

        if peer_url not in self.peer_data_sockets:
            assert self.data_socket is not None, "Data socket not initialized"
            loop = asyncio.get_event_loop()
            conn, _ = await loop.sock_accept(self.data_socket)
            self.peer_data_sockets[peer_url] = conn

        sock = self.peer_data_sockets[peer_url]

        meta_bytes = await self._async_recv_data(sock)
        meta_msg = msgspec.msgpack.decode(meta_bytes, type=DataTransferMsg)

        assert meta_msg.num_objects == len(buffers)

        for i, buffer in enumerate(buffers):
            data = await self._async_recv_data(sock)
            if isinstance(buffer, MemoryObj):
                # Standard
                import ctypes

                ctypes.memmove(buffer.data_ptr, data, len(data))
            elif isinstance(buffer, bytearray):
                buffer[:] = data
            else:
                raise ValueError(f"Unsupported buffer type: {type(buffer)}")

        return len(buffers)

    def batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Write operation is same as send for socket channel"""
        return self.batched_send(objects, transfer_spec)

    def batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Read operation is same as recv for socket channel"""
        return self.batched_recv(buffers, transfer_spec)

    async def async_batched_write(
        self,
        objects: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async write operation is same as async send for socket channel"""
        return await self.async_batched_send(objects, transfer_spec)

    async def async_batched_read(
        self,
        buffers: Union[list[bytes], list[MemoryObj]],
        transfer_spec: Optional[dict] = None,
    ) -> int:
        """Async read operation is same as async recv for socket channel"""
        return await self.async_batched_recv(buffers, transfer_spec)

    def close(self):
        """Close all sockets and cleanup resources"""
        super().close()

        for sock in self.peer_data_sockets.values():
            sock.close()

        if self.data_socket is not None:
            self.data_socket.close()
