# SPDX-License-Identifier: Apache-2.0
"""Re-export the common ``/loglevel`` endpoint."""

# First Party
from lmcache.v1.internal_api_server.common.loglevel_api import (
    router,
)

__all__ = ["router"]
