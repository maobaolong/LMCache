# SPDX-License-Identifier: Apache-2.0
"""Select a compatible official nightly without comparing it with PyPI releases."""

# Standard
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen
import argparse
import re

# Third Party
from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import Version


@dataclass(frozen=True)
class NightlySelection:
    """Installed-version constraint and immutable CUDA archive coordinates."""

    version: str
    source_sha: str
    archive_index_url: str


class WheelLinks(HTMLParser):
    """Collect wheel destinations from a PEP 503 index."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect anchor hrefs; ignore all other index markup."""
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value is not None:
                    self.links.append(value)


def select_nightly(index_url: str, cuda_variant: str) -> NightlySelection:
    """Resolve the newest compatible wheel listed by the official nightly index.

    Args:
        index_url: PEP 503 vLLM package-index URL, including its trailing slash.
        cuda_variant: The cuNNN archive variant validated by the shell resolver.

    Returns:
        The exact version, full source SHA and immutable CUDA archive URL.

    Raises:
        ValueError: No compatible wheel exists, or its source coordinates do
            not match the wheel's git-local version suffix.
        urllib.error.URLError: The index cannot be fetched.
    """
    with urlopen(index_url, timeout=20) as response:
        page = response.read().decode("utf-8")
    parser = WheelLinks()
    parser.feed(page)
    supported = set(sys_tags())
    candidates: list[tuple[Version, str]] = []
    for link in parser.links:
        wheel_url = urljoin(index_url, link)
        filename = unquote(PurePosixPath(urlsplit(wheel_url).path).name)
        try:
            name, version, _, tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if name == "vllm" and supported.intersection(tags):
            candidates.append((version, wheel_url))
    if not candidates:
        raise ValueError(f"No compatible vLLM nightly wheel in {index_url}")
    version, wheel_url = max(candidates, key=lambda item: item[0])
    source_sha = next(
        (
            part
            for part in urlsplit(wheel_url).path.split("/")
            if re.fullmatch(r"[0-9a-f]{40}", part)
        ),
        "",
    )
    if (
        not source_sha
        or version.local is None
        or not version.local.startswith("g")
        or not source_sha.startswith(version.local[1:])
    ):
        raise ValueError(
            f"Nightly wheel has missing or inconsistent source SHA: {wheel_url}"
        )
    return NightlySelection(
        str(version), source_sha, f"https://wheels.vllm.ai/{source_sha}/{cuda_variant}"
    )


def main() -> None:
    """Print version, source SHA and archive URL as one tab-separated record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_url")
    parser.add_argument("cuda_variant")
    args = parser.parse_args()
    selected = select_nightly(args.index_url, args.cuda_variant)
    print(selected.version, selected.source_sha, selected.archive_index_url, sep="\t")


if __name__ == "__main__":
    main()
