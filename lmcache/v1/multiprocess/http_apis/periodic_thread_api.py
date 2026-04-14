# SPDX-License-Identifier: Apache-2.0
"""Re-export the common periodic-thread endpoints."""

# First Party
from lmcache.v1.internal_api_server.common.periodic_thread_api import (
    router,
)

__all__ = ["router"]
