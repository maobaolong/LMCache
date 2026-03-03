# SPDX-License-Identifier: Apache-2.0
"""
Tests for MooncakestoreConnector ping/support_ping functionality.

Uses mock store to verify connector behavior without real Mooncake dependency.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest


class TestMooncakeConnectorPing:
    """Test support_ping and ping on MooncakestoreConnector."""

    def _make_connector_with_mock_store(
        self, health_check_return=0, has_health_check=True
    ):
        """Create a MooncakestoreConnector with a mocked store.

        We patch the __init__ to skip real Mooncake setup,
        then manually set the store mock.
        """
        # First Party
        from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
            MooncakestoreConnector,
        )

        mock_store = MagicMock()
        if has_health_check:
            mock_store.health_check.return_value = health_check_return
        else:
            del mock_store.health_check

        # Create connector bypassing __init__
        connector = object.__new__(MooncakestoreConnector)
        connector.store = mock_store
        return connector, mock_store

    def test_support_ping_returns_true(self):
        """support_ping() should return True when store has health_check."""
        connector, _ = self._make_connector_with_mock_store()
        assert connector.support_ping() is True

    def test_support_ping_returns_false_without_health_check(self):
        """support_ping() should return False when store lacks health_check."""
        connector, _ = self._make_connector_with_mock_store(has_health_check=False)
        assert connector.support_ping() is False

    @pytest.mark.asyncio
    async def test_ping_returns_zero_without_health_check(self):
        """ping() should return 0 when store lacks health_check."""
        connector, _ = self._make_connector_with_mock_store(has_health_check=False)
        result = await connector.ping()
        assert result == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "health_check_return, description",
        [
            (0, "healthy"),
            (1, "not initialized"),
            (2, "master unreachable"),
        ],
    )
    async def test_ping_forwards_health_check_status(
        self, health_check_return, description
    ):
        """ping() should forward the store.health_check() return value."""
        connector, mock_store = self._make_connector_with_mock_store(
            health_check_return=health_check_return
        )
        result = await connector.ping()
        assert result == health_check_return, (
            f"Expected {health_check_return} ({description}), got {result}"
        )
        mock_store.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_ping_propagates_exception(self):
        """ping() should propagate exceptions from store.health_check()."""
        connector, mock_store = self._make_connector_with_mock_store()
        mock_store.health_check.side_effect = RuntimeError("connection lost")

        with pytest.raises(RuntimeError, match="connection lost"):
            await connector.ping()
