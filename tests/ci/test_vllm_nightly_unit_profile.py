# SPDX-License-Identifier: Apache-2.0
"""Public unit-entrypoint contracts for installed-runtime qualification."""

# Standard
from pathlib import Path
import os
import shlex
import subprocess
import sys

# Third Party
import pytest

# First Party
from tests.import_utils import repo_path

VERSION = "0.34.0rc1.dev1+g0123456"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("nightly", "installed_version", "startup_hook", "succeeds"),
    [
        (False, VERSION, False, True),
        (True, VERSION, True, True),
        (True, "0.34.0", True, False),
        (True, VERSION, False, False),
    ],
    ids=["default-no-runtime", "qualified-nightly", "wrong-version", "missing-hook"],
)
def test_unit_entrypoint_qualifies_runtime_before_pytest(
    tmp_path: Path,
    nightly: bool,
    installed_version: str,
    startup_hook: bool,
    succeeds: bool,
) -> None:
    """An opt-in profile requires a matching installed runtime before tests."""
    root = tmp_path / "repo"
    entrypoint = root / ".buildkite/k3_tests/unit/run.sh"
    _write(entrypoint, repo_path(".buildkite/k3_tests/unit/run.sh").read_text())
    for name, profile in [
        ("setup-env.sh", "nightly"),
        ("setup-lmcache-only-env.sh", "standard"),
    ]:
        _write(
            root / ".buildkite/k3_harness" / name,
            f'printf "{profile}\\n" > "$TEST_SETUP_LOG"\n'
            f"export PINNED_VLLM_VERSION={shlex.quote(VERSION)}\n",
        )
    _write(root / "lmcache/v1/transport/grpc/_proto_gen/_generate.py", "pass\n")

    modules = tmp_path / "modules"
    package = modules / "vllm"
    for directory in [
        "",
        "distributed",
        "distributed/kv_transfer",
        "distributed/kv_transfer/kv_connector",
        "distributed/kv_transfer/kv_connector/v1",
    ]:
        _write(package / directory / "__init__.py", "")
    _write(
        package / "__init__.py",
        f"__version__ = {installed_version!r}\n"
        if nightly
        else "raise RuntimeError('default units must not import vllm')\n",
    )
    _write(
        package / "distributed/kv_transfer/kv_connector/v1/base.py",
        "class KVConnectorBase_V1:\n"
        + ("    def on_kv_cache_ready(self): pass\n" if startup_hook else "    pass\n"),
    )
    _write(
        modules / f"vllm-{installed_version}.dist-info/METADATA",
        f"Metadata-Version: 2.1\nName: vllm\nVersion: {installed_version}\n",
    )

    binaries = tmp_path / "bin"
    _write(binaries / "python", f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    for name in ["uv", "mkdir", "rm"]:
        _write(binaries / name, "#!/bin/sh\nexit 0\n")
    _write(
        binaries / "pytest", '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_PYTEST_ARGS"\n'
    )
    _write(binaries / "buildkite-agent", "#!/bin/sh\ncat >/dev/null\n")
    setup_log, pytest_args = tmp_path / "setup.txt", tmp_path / "pytest.txt"
    env = dict(
        os.environ,
        PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
        PYTHONPATH=str(modules),
        BUILDKITE_BUILD_ID="unit-profile-fixture",
        CI_WEEKLY_FULL="0",
        VLLM_NIGHTLY_UNIT_TESTS="1" if nightly else "0",
        TEST_SETUP_LOG=str(setup_log),
        TEST_PYTEST_ARGS=str(pytest_args),
    )
    result = subprocess.run(
        ["bash", str(entrypoint)], env=env, text=True, capture_output=True, timeout=20
    )
    assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
    assert setup_log.read_text().strip() == ("nightly" if nightly else "standard")
    assert pytest_args.exists() is succeeds
    if succeeds:
        args = pytest_args.read_text().splitlines()
        assert ("--junitxml=junit-unit.xml" in args) is nightly
        assert ("--maxfail=1" in args) is not nightly
    else:
        expected = (
            "exact selected runtime"
            if installed_version != VERSION
            else "startup-ready callback"
        )
        assert expected in result.stderr
