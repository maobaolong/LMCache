# SPDX-License-Identifier: Apache-2.0
"""gRPC adapter for the generated ``DemoService`` surface."""

# First Party
from lmcache.v1.multiprocess.modules.demo import DemoModule


class DemoServiceImpl:
    """Implementation of the generated ``DemoService`` RPC surface."""

    def __init__(self, demo: DemoModule) -> None:
        self._demo = demo

    def Demo(self, demo_value: str) -> str:
        """Return a demo response value."""
        return self._demo.demo(demo_value)
