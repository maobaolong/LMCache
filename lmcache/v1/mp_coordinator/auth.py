# SPDX-License-Identifier: Apache-2.0
"""Explicitly configured service credentials for fleet HTTP requests."""

# First Party
from lmcache.security.http import bearer_authorization_header
from lmcache.v1.multiprocess.transport.auth import load_request_client_api_key


def fleet_auth_headers() -> dict[str, str]:
    """Load the single sending credential for coordinator/MP HTTP clients.

    The shared service environment supplies one key. Accepted server keys may
    overlap during rotation, but clients must select one source. Credentials
    are loaded locally, never forwarded from an incoming HTTP request.

    Returns:
        An Authorization header, or no headers when no key is configured.

    Raises:
        OSError: If the configured key file cannot be read.
        ValueError: If multiple sending sources or an invalid key file are configured.
    """
    api_key = load_request_client_api_key()
    return (
        {"Authorization": bearer_authorization_header(api_key)}
        if api_key is not None
        else {}
    )
