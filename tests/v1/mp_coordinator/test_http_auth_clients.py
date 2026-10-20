# SPDX-License-Identifier: Apache-2.0
"""Fleet credentials on real HTTP client requests, using in-memory transports."""

# Standard
from collections.abc import Callable, Iterator
from http.client import HTTPMessage
from io import BytesIO
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock
import asyncio
import urllib.request

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
import uvicorn

# First Party
from lmcache.cli.commands.query._coordinator import get_json
from lmcache.cli.commands.quota._helpers import http_request
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.mp_coordinator.api import BlendNamespace
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient
from lmcache.v1.mp_coordinator.cache_events import HttpCacheEventSink
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.registrar import bind_coordinator_chunk_size
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess import http_server
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    HTTPFrontendConfig,
    MPServerConfig,
    P2PConfig,
)
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController

KEY = "fleet-test-key"


@pytest.fixture(params=["none", "value", "file", "legacy", "precedence"])
def credential(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> str | None:
    """Exercise the same sending sources accepted by the shared request adapter."""
    for name in (
        "LMCACHE_API_KEY",
        "LMCACHE_API_KEY_FILE",
        "LMCACHE_HTTP_API_KEY",
        "LMCACHE_HTTP_API_KEY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    if request.param == "value":
        monkeypatch.setenv("LMCACHE_API_KEY", KEY)
    elif request.param == "file":
        path = tmp_path / "fleet.key"
        path.write_text(KEY)
        monkeypatch.setenv("LMCACHE_API_KEY_FILE", str(path))
    elif request.param == "legacy":
        monkeypatch.setenv("LMCACHE_HTTP_API_KEY", KEY)
    elif request.param == "precedence":
        monkeypatch.setenv("LMCACHE_API_KEY", KEY)
        monkeypatch.setenv("LMCACHE_HTTP_API_KEY", "old-key")
    return None if request.param == "none" else f"Bearer {KEY}"


@pytest.fixture
def sync_requests(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[httpx.Request]]:
    """Capture real sync HTTP requests while retaining constructor arguments."""
    captured: list[httpx.Request] = []
    real_client = httpx.Client

    def respond(request: httpx.Request) -> httpx.Response:
        """Record one request and return harmless coordinator response shapes."""
        captured.append(request)
        return httpx.Response(
            200, json={"matches": [], "instances": [], "chunk_size": 256}
        )

    def client(
        *, timeout: float = 2.0, headers: dict[str, str] | None = None
    ) -> httpx.Client:
        """Keep production headers and replace only network I/O."""
        return real_client(
            transport=httpx.MockTransport(respond), timeout=timeout, headers=headers
        )

    monkeypatch.setattr(httpx, "Client", client)
    yield captured


def test_event_sink_sends_the_configured_key(
    credential: str | None, sync_requests: list[httpx.Request]
) -> None:
    """The independent event client must authenticate its POST /events."""
    sink = HttpCacheEventSink("http://coordinator")
    try:
        sink.publish([])
    finally:
        sink.close()
    assert len(sync_requests) == 1
    assert sync_requests[0].headers.get("authorization") == credential


def test_blend_lookup_sends_the_configured_key(
    credential: str | None, sync_requests: list[httpx.Request]
) -> None:
    """Background CacheBlend calls must use the same fleet sending key."""
    client = BlendCoordinatorClient("http://coordinator")
    try:
        client.submit_match(
            "request", [1, 2], BlendNamespace(model_name="model", world_size=1)
        )
        # A public completion poll avoids inspecting the client's queue/thread.
        done = Event()
        for _ in range(100):
            result = client.poll_match("request")
            if isinstance(result, list):
                break
            done.wait(0.01)
        assert result == []
    finally:
        client.close()
    assert len(sync_requests) == 1
    assert sync_requests[0].headers.get("authorization") == credential


def test_p2p_discovery_sends_the_configured_key(
    credential: str | None,
    sync_requests: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke the discovery callback supplied to the public timer factory."""
    callbacks: list[Callable[[], object]] = []

    def timer(*, execute_fn: Callable[[], object], **kwargs: object) -> MagicMock:
        """Capture scheduling without starting a network poll thread."""
        callbacks.append(execute_fn)
        return MagicMock()

    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.p2p_controller.initialize_transfer_channel_context",
        MagicMock(),
    )
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.p2p_controller.create_periodic_thread", timer
    )
    controller = P2PController(
        MagicMock(),
        P2PConfig(advertise_url="127.0.0.1:7600"),
        CoordinatorConfig(url="http://coordinator"),
        "node",
    )
    try:
        callbacks[0]()
    finally:
        controller.close()
    assert len(sync_requests) == 1
    assert sync_requests[0].headers.get("authorization") == credential


def test_chunk_size_negotiation_sends_the_configured_key(
    credential: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The synchronous startup negotiation is a separate authenticated path."""
    captured: list[dict[str, str]] = []

    def put(
        url: str,
        *,
        json: dict[str, int],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Capture the real helper's HTTP call without opening a socket."""
        captured.append(headers or {})
        return httpx.Response(
            200, json={"chunk_size": 256}, request=httpx.Request("PUT", url)
        )

    monkeypatch.setattr(httpx, "put", put)
    assert bind_coordinator_chunk_size("http://coordinator", 256) == 256
    assert captured[0].get("Authorization") == credential


@pytest.mark.parametrize("command", ["query", "quota"])
def test_cli_sends_credentials_without_forwarding_them_on_redirect(
    credential: str | None, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """urllib must send the key only to the operator's configured destination."""
    captured: list[urllib.request.Request] = []

    def open_request(request: urllib.request.Request, **kwargs: object) -> BytesIO:
        """Capture the initial request, preserving normal urllib headers."""
        captured.append(request)
        return BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    if command == "query":
        assert get_json("http://coordinator/instances") == {}
    else:
        assert http_request("GET", "http://coordinator/quota") == {}
    request = captured[0]
    assert request.get_header("Authorization") == credential
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        request, BytesIO(), 302, "Found", HTTPMessage(), "http://other-service/"
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def test_coordinator_outbound_client_uses_its_own_key(
    credential: str | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An operator's key must not replace the coordinator's MP sending key."""
    captured: list[str | None] = []
    real_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        """Capture the actual warm-prefetch dispatch to the registered MP."""
        captured.append(request.headers.get("authorization"))
        return httpx.Response(
            202, json={"request_id": "prefetch", "chunks": 1, "status": "submitted"}
        )

    def client(
        *, timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> httpx.AsyncClient:
        """Keep production headers and replace only network I/O."""
        return real_client(
            transport=httpx.MockTransport(respond), timeout=timeout, headers=headers
        )

    operator_key = tmp_path / "operator.key"
    operator_key.write_text("operator-test-key")
    monkeypatch.setattr(httpx, "AsyncClient", client)
    app = create_app(
        MPCoordinatorConfig(
            api_key_files=(str(operator_key),),
            health_check_interval=0,
            eviction_check_interval=0,
            metrics_enabled=False,
        )
    )
    headers = {"Authorization": "Bearer operator-test-key"}
    with TestClient(app) as channel:
        assert (
            channel.post(
                "/instances",
                json={"instance_id": "node", "ip": "127.0.0.1", "http_port": 8080},
                headers=headers,
            ).status_code
            == 200
        )
        response = channel.post(
            "/cache/prefetches",
            json={
                "instance_id": "node",
                "model_name": "model",
                "world_size": 1,
                "token_ids": [1, 2, 3, 4],
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
    assert captured == [credential]


def test_mp_lifespan_authenticates_registration_and_deregistration(
    credential: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public MP launcher must join and leave a keyed coordinator."""
    coordinator = create_app(MPCoordinatorConfig(metrics_enabled=False))
    transport = httpx.ASGITransport(app=coordinator)
    captured: list[tuple[str, str, str | None]] = []
    real_client = httpx.AsyncClient

    async def relay(request: httpx.Request) -> httpx.Response:
        """Send actual registration requests to the in-process coordinator."""
        captured.append(
            (request.method, request.url.path, request.headers.get("authorization"))
        )
        return await transport.handle_async_request(request)

    def client(
        *, timeout: float, headers: dict[str, str] | None = None
    ) -> httpx.AsyncClient:
        """Keep production headers while replacing the network transport."""
        return real_client(
            timeout=timeout, headers=headers, transport=httpx.MockTransport(relay)
        )

    class Server:
        """Run the real app lifespan without binding a listening socket."""

        def __init__(self, config: uvicorn.Config) -> None:
            self.config = config

        def run(self) -> None:
            """Start registration and allow normal shutdown to deregister."""
            app = self.config.app
            assert isinstance(app, FastAPI)

            async def lifecycle() -> None:
                async with app.router.lifespan_context(app):
                    await asyncio.sleep(0)

            asyncio.run(lifecycle())

    engine = MagicMock()
    engine.context.finalized_chunk_size = None
    monkeypatch.setattr(
        http_server, "run_cache_server", MagicMock(return_value=(MagicMock(), engine))
    )
    monkeypatch.setattr(http_server, "build_context", MagicMock())
    monkeypatch.setattr(
        http_server, "get_event_bus", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    http_server.run_http_server(
        HTTPFrontendConfig(),
        MPServerConfig(instance_id="node"),
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(size_in_bytes=65536, use_lazy=False)
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        ),
        ObservabilityConfig(enabled=False),
        CoordinatorConfig(url="http://coordinator", advertise_ip="127.0.0.1"),
    )
    assert captured == [
        ("POST", "/instances", credential),
        ("DELETE", "/instances/node", credential),
    ]


def test_ambiguous_sending_credentials_are_rejected(
    credential: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clients must not arbitrarily choose one of multiple accepted keys."""
    path = tmp_path / "second.key"
    path.write_text("second-key")
    monkeypatch.setenv("LMCACHE_API_KEY", KEY)
    monkeypatch.setenv("LMCACHE_API_KEY_FILE", str(path))
    with pytest.raises(ValueError, match="exactly one"):
        HttpCacheEventSink("http://coordinator")
