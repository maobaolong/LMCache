# SPDX-License-Identifier: Apache-2.0
"""Demo module used by the gRPC service expansion example."""


class DemoModule:
    """Communication-agnostic demo logic used by the expansion example."""

    def demo(self, demo_value: str) -> str:
        """Return a demo response value."""
        return f"demo:{demo_value}"
