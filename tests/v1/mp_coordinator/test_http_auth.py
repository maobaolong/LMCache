# SPDX-License-Identifier: Apache-2.0
"""Coordinator admission through the public HTTP and CLI interfaces."""

# Standard
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import argparse

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.cli.commands.coordinator import CoordinatorCommand
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

KEY = "coordinator-test-key"


@pytest.fixture(autouse=True)
def clean_key_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the operator's actual service credentials."""
    for name in (
        "LMCACHE_API_KEY",
        "LMCACHE_API_KEY_FILE",
        "LMCACHE_HTTP_API_KEY",
        "LMCACHE_HTTP_API_KEY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Run a keyed coordinator without network work or periodic controllers."""
    monkeypatch.setenv("LMCACHE_API_KEY", KEY)
    config = MPCoordinatorConfig(
        health_check_interval=0, eviction_check_interval=0, metrics_enabled=False
    )
    with TestClient(create_app(config)) as channel:
        yield channel


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic wrong"])
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/instances"),
        ("POST", "/instances"),
        ("PUT", "/instances/node/heartbeat"),
        ("DELETE", "/instances/node"),
        ("PUT", "/config/chunk-size"),
        ("POST", "/events"),
        ("GET", "/directory/stats"),
        ("POST", "/directory/blend-lookup"),
        ("GET", "/quota"),
        ("PUT", "/quota/tenant"),
        ("POST", "/cache/pins"),
        ("POST", "/cache/prefetches"),
        ("POST", "/cache/delete"),
        ("GET", "/metrics"),
        ("GET", "/docs"),
        ("GET", "/openapi.json"),
    ],
)
def test_routes_require_a_bearer_key(
    client: TestClient, method: str, path: str, authorization: str | None
) -> None:
    """Authentication must precede schema validation and every route handler."""
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.request(method, path, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_rejected_mutations_leave_membership_unchanged(client: TestClient) -> None:
    """Wrong-key register/delete requests must not mutate the fleet registry."""
    body = {"instance_id": "node", "ip": "127.0.0.1", "http_port": 8080}
    headers = {"Authorization": f"Bearer {KEY}"}
    assert client.post("/instances", json=body).status_code == 401
    assert client.get("/instances", headers=headers).json()["instances"] == []
    assert client.post("/instances", json=body, headers=headers).status_code == 200
    assert client.delete("/instances/node").status_code == 401
    nodes = client.get("/instances", headers=headers).json()["instances"]
    assert [node["instance_id"] for node in nodes] == ["node"]
    assert client.delete("/instances/node", headers=headers).status_code == 204


def test_health_probe_is_public(client: TestClient) -> None:
    """The exact liveness path stays available without credentials."""
    assert client.get("/healthz").json() == {"status": "healthy"}
    assert client.get("/healthz/").status_code == 401


def test_routes_added_after_creation_are_protected(client: TestClient) -> None:
    """App-wide admission also covers endpoints supplied by extensions."""

    app = client.app
    assert isinstance(app, FastAPI)

    @app.get("/extension")
    def extension() -> dict[str, bool]:
        """Return the extension's harmless test payload."""
        return {"ok": True}

    assert client.get("/extension").status_code == 401
    assert client.get(
        "/extension", headers={"Authorization": f"Bearer {KEY}"}
    ).json() == {"ok": True}


def test_unconfigured_coordinator_keeps_existing_behavior() -> None:
    """No configured keys preserves the existing unauthenticated deployment."""
    with TestClient(create_app(MPCoordinatorConfig(metrics_enabled=False))) as channel:
        assert channel.get("/instances").status_code == 200
        assert channel.get("/docs").status_code == 200


def test_cli_key_files_support_rotation_without_exporting_keys(tmp_path: Path) -> None:
    """Both CLI spellings feed the shared loader; config contains paths only."""
    paths = [tmp_path / "old.key", tmp_path / "new.key"]
    for path, value in zip(paths, ["old-key", "new-key"], strict=True):
        path.write_text(value)
    parser = argparse.ArgumentParser()
    command = CoordinatorCommand()
    command.add_arguments(parser)
    args = parser.parse_args(
        [
            "--api-key-file",
            str(paths[0]),
            "--http-api-key-file",
            str(paths[1]),
            "--disable-metrics",
        ]
    )
    with patch("uvicorn.run") as serve:
        command.execute(args)
    app = serve.call_args.args[0]
    config = app.state.config
    assert config.api_key_files == tuple(map(str, paths))
    exported = str(asdict(config))
    assert "old-key" not in exported and "new-key" not in exported
    with TestClient(app) as channel:
        for token in ["old-key", "new-key"]:
            assert (
                channel.get(
                    "/instances", headers={"Authorization": f"Bearer {token}"}
                ).status_code
                == 200
            )
        assert channel.get("/instances").status_code == 401


@pytest.mark.parametrize("value", ["", "one\ntwo\n"])
def test_invalid_key_files_fail_before_startup(tmp_path: Path, value: str) -> None:
    """A bad configured file must not silently disable authentication."""
    path = tmp_path / "invalid.key"
    path.write_text(value)
    with pytest.raises(ValueError, match="API key file"):
        create_app(MPCoordinatorConfig(api_key_files=(str(path),)))
