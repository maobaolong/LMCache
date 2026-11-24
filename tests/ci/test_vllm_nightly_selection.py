# SPDX-License-Identifier: Apache-2.0
"""Exercise the nightly selector CLI without installing an engine or using GPUs."""

# Standard
from pathlib import Path
from urllib.parse import quote
import subprocess
import sys

# Third Party
from packaging.tags import sys_tags
import pytest

# First Party
from tests.import_utils import repo_path

SCRIPT = repo_path(".buildkite", "k3_harness", "select-vllm-nightly.py")
SOURCE = "e04f069197ee4b3b5a6c67932735edb10d4e4ba2"
VERSION = "0.34.0rc1.dev1216+ge04f06919"


def wheel_link(version: str, source: str, tag: str) -> str:
    """Create an index anchor with a wheel filename and source directory."""
    filename = quote(f"vllm-{version}-{tag}.whl")
    return f'<a href="{source}/{filename}">wheel</a>'


def select(
    tmp_path: Path, links: list[str], variant: str = "cu132"
) -> subprocess.CompletedProcess[str]:
    """Run the public selector CLI against a local PEP 503 index fixture."""
    index = tmp_path / "index.html"
    index.write_text("\n".join(links))
    return subprocess.run(
        [sys.executable, str(SCRIPT), index.as_uri(), variant],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("variant", ["cu130", "cu132"])
def test_selects_compatible_official_nightly_and_immutable_archive(
    tmp_path: Path, variant: str
) -> None:
    """Use the nightly index's compatible wheel even if PyPI has a higher release."""
    native = str(next(sys_tags()))
    result = select(
        tmp_path,
        [
            wheel_link(
                "0.34.0rc1.dev957+gb2c92fa9c",
                "b2c92fa9c1d8be6790ddd0852b547bbf927bfbd0",
                native,
            ),
            wheel_link(VERSION, SOURCE, native),
            wheel_link(
                "99.0.dev1+gfffffffff", "f" * 40, "cp38-abi3-unsupported_platform"
            ),
        ],
        variant,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split("\t") == [
        VERSION,
        SOURCE,
        f"https://wheels.vllm.ai/{SOURCE}/{variant}",
    ]


def test_no_compatible_nightly_fails_instead_of_falling_back_to_pypi(
    tmp_path: Path,
) -> None:
    """A missing nightly cannot silently become an older stable runtime."""
    result = select(
        tmp_path, [wheel_link(VERSION, SOURCE, "cp38-abi3-unsupported_platform")]
    )
    assert result.returncode != 0
    assert "No compatible vLLM nightly wheel" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("source", ["not-a-source-sha", "a" * 40])
def test_missing_or_inconsistent_source_fails_before_installation(
    tmp_path: Path, source: str
) -> None:
    """A selected wheel must identify the same immutable source as its version."""
    result = select(tmp_path, [wheel_link(VERSION, source, str(next(sys_tags())))])
    assert result.returncode != 0
    assert "source SHA" in result.stderr
    assert not result.stdout
