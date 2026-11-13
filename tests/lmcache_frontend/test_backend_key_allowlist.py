# SPDX-License-Identifier: Apache-2.0
"""Backend credential authorization tests for the LMCache frontend."""

# Standard
from collections.abc import Iterator
import asyncio
import json
import sys

# Third Party
from fastapi.testclient import TestClient
import httpx
import pytest

# First Party
from lmcache.lmcache_frontend import app as frontend
from lmcache.security.api_key import APIKeyVerifier

TARGET = "127.0.0.1:18085"
BACKEND_AUTHORIZATION = "Bearer backend-secret"


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset process-global frontend configuration around each test."""
    frontend.target_nodes.clear()
    monkeypatch.setattr(frontend, "args", None)
    yield
    frontend.target_nodes.clear()


@pytest.mark.parametrize("allowlisted", [False, True])
def test_supplier_target_receives_backend_key_only_when_allowlisted(
    monkeypatch: pytest.MonkeyPatch, allowlisted: bool
) -> None:
    """Unauthenticated supplier data cannot choose a backend-key recipient."""
    captured_authorization: list[str | None] = []

    def backend(request: httpx.Request) -> httpx.Response:
        if request.url.host == "supplier":
            return httpx.Response(
                200,
                json={
                    "processInfos": {
                        "worker": {
                            "lmCacheInfoEntities": [{"apiAddress": f"http://{TARGET}"}]
                        }
                    }
                },
            )
        captured_authorization.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"nodes": []})

    async_client = httpx.AsyncClient

    def make_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(backend)
        return async_client(**kwargs)

    monkeypatch.setattr(frontend.httpx, "AsyncClient", make_client)
    allowed = {TARGET} if allowlisted else set()
    nodes = asyncio.run(
        frontend.fetch_nodes_from_supplier(
            "http://supplier/lmcache_infos",
            BACKEND_AUTHORIZATION,
            allowed,
        )
    )

    assert nodes
    assert captured_authorization == [BACKEND_AUTHORIZATION if allowlisted else None]


@pytest.mark.parametrize("route", ["proxy", "refresh", "metrics"])
@pytest.mark.parametrize("allowlisted", [False, True])
def test_discovered_target_uses_backend_key_only_when_allowlisted(
    monkeypatch: pytest.MonkeyPatch, route: str, allowlisted: bool
) -> None:
    """Every outbound TCP route applies the same operator trust boundary."""
    captured_authorization: list[str | None] = []

    def backend(request: httpx.Request) -> httpx.Response:
        captured_authorization.append(request.headers.get("authorization"))
        if request.url.path == "/api/nodes":
            return httpx.Response(200, json={"nodes": []})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="cache_hits 1\n")
        return httpx.Response(200, text="ok")

    async_client = httpx.AsyncClient

    def make_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(backend)
        return async_client(**kwargs)

    monkeypatch.setattr(frontend.httpx, "AsyncClient", make_client)
    host, port = TARGET.split(":")
    frontend.target_nodes.append(
        {
            "name": "proxy",
            "host": host,
            "port": port,
            "nodes": [{"name": "node", "host": host, "port": port}],
        }
    )
    allowed = {TARGET} if allowlisted else set()
    app = frontend.create_app(
        APIKeyVerifier(["frontend-secret"]),
        BACKEND_AUTHORIZATION,
        allowed_backend_key_targets=allowed,
    )

    with TestClient(app) as client:
        if route == "proxy":
            response = client.get(
                "/proxy2/node/status",
                headers={"Authorization": "Bearer frontend-secret"},
            )
        elif route == "refresh":
            response = client.get(
                "/api/proxies/proxy/refresh",
                headers={"Authorization": "Bearer frontend-secret"},
            )
        else:
            response = client.get("/metrics")

    assert response.status_code == 200
    assert captured_authorization == [BACKEND_AUTHORIZATION if allowlisted else None]


def test_main_trusts_operator_configured_tcp_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Targets declared through --nodes keep authenticated backend behavior."""
    captured_apps = []
    nodes = [{"name": "node", "host": "127.0.0.1", "port": "18085"}]
    for name in (
        "LMCACHE_API_KEY",
        "LMCACHE_API_KEY_FILE",
        "LMCACHE_HTTP_API_KEY",
        "LMCACHE_HTTP_API_KEY_FILE",
        "LMCACHE_FRONTEND_BACKEND_API_KEY",
        "LMCACHE_FRONTEND_BACKEND_API_KEY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lmcache-frontend", "--nodes", json.dumps(nodes)],
    )
    monkeypatch.setenv("LMCACHE_API_KEY", "frontend-secret")
    monkeypatch.setenv("LMCACHE_FRONTEND_BACKEND_API_KEY", "backend-secret")
    monkeypatch.setattr(
        frontend.uvicorn,
        "run",
        lambda app, **_kwargs: captured_apps.append(app),
    )

    frontend.main()

    assert len(captured_apps) == 1
    assert TARGET in captured_apps[0].state.allowed_backend_key_targets
