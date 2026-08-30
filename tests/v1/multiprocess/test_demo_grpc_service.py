# SPDX-License-Identifier: Apache-2.0
"""Expansion example for adding an independent gRPC service."""

# Standard
import socket

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.modules.demo import DemoModule
from lmcache.v1.multiprocess.transport.grpc_impl.grpc import (
    MultiprocessGrpcClient,
    MultiprocessGrpcServer,
)
from lmcache.v1.multiprocess.transport.grpc_impl.protocol import RPC
from lmcache.v1.multiprocess.transport.grpc_impl.services.demo import DemoServiceImpl


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_demo_service_new_rpc_roundtrip() -> None:
    """A new service RPC is callable without transport registry edits."""
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"
    server = MultiprocessGrpcServer(server_url)
    server.add_service("DemoService", DemoServiceImpl(DemoModule()))
    server.start()
    client = MultiprocessGrpcClient(server_url)

    try:
        assert RPC.Demo.service_name == "DemoService"
        future: MessagingFuture[str] = client.demo("request-value")
        assert future.result(timeout=5.0) == "demo:request-value"
    finally:
        client.close()
        server.close()
