# SPDX-License-Identifier: Apache-2.0
"""
LMCache Standalone Module

This module provides standalone starter for LMCacheEngine that works
without vLLM or GPU.
"""

# Local
from .lmcache_standalone import LMCacheStandaloneStarter

__all__ = ["LMCacheStandaloneStarter"]
