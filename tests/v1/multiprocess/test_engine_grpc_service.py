# SPDX-License-Identifier: Apache-2.0
"""Expansion example for adding an RPC to EngineService."""

# Standard
from unittest.mock import MagicMock
import socket

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.grpc_impl.grpc import (
    MultiprocessGrpcClient,
    MultiprocessGrpcServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.protocol import RPC
from lmcache.v1.multiprocess.transport.grpc_impl.services.engine import (
    EngineServiceImpl,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_engine_service_new_rpc_roundtrip() -> None:
    """A new EngineService RPC is callable without transport registry edits."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    engine_service = EngineServiceImpl(MagicMock())
    server = MultiprocessGrpcServer(server_url)
    server.add_service("EngineService", engine_service)
    server.assign_thread_pools(max_cpu_workers=2, max_gpu_workers=2)
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert RPC.EchoEngineProbe.service_name == "EngineService"
        future: MessagingFuture[str] = client.echo_engine_probe("engine-probe")
        assert future.result(timeout=5.0) == "engine-probe"
    finally:
        client.close()
        server.close()
