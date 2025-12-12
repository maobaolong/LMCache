# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import hashlib
import json

# Third Party
from fastapi.testclient import TestClient
import pytest
import torch

# First Party
from lmcache.v1.internal_api_server.api_server import app


class TestKVCacheCheckAPI:
    """Test suite for the /kvcache/check API endpoint."""

    @pytest.fixture
    def mock_kv_caches(self) -> Dict[str, torch.Tensor]:
        """Create mock kv_caches tensors for testing."""
        # Create kv_caches with shape [num_blocks, block_size, num_heads, head_dim]
        # Total slots = num_blocks * block_size = 4 * 4 = 16
        kv_caches = {
            "layer_0": torch.randn(4, 4, 8, 64),  # [4, 4, 8, 64]
            "layer_1": torch.randn(4, 4, 8, 64),
        }
        return kv_caches

    @pytest.fixture
    def mock_lmcache_adapter(self, mock_kv_caches):
        """Create a mock LMCacheConnectorV1Impl adapter."""
        adapter = MagicMock()
        adapter.kv_caches = mock_kv_caches

        def compute_checksums(
            slot_indices: List[int], chunk_size: Optional[int] = None
        ) -> Optional[Dict[str, Any]]:
            """Mock implementation of compute_kvcache_checksums."""
            if not adapter.kv_caches:
                return None

            layer_checksums: Dict[str, str] = {}
            chunk_checksums: Dict[str, List[str]] = {}

            for layer_name, kv_tensor in adapter.kv_caches.items():
                slot_tensor = torch.tensor(
                    slot_indices, dtype=torch.long, device=kv_tensor.device
                )
                kv_at_slots = kv_tensor.view(-1, *kv_tensor.shape[2:])[slot_tensor]

                # Compute overall layer checksum
                tensor_bytes = kv_at_slots.detach().cpu().contiguous().numpy().tobytes()
                checksum = hashlib.md5(tensor_bytes).hexdigest()
                layer_checksums[layer_name] = checksum

                # Compute per-chunk checksums if chunk_size is provided
                if chunk_size is not None and chunk_size > 0:
                    num_slots = len(slot_indices)
                    num_chunks = (num_slots + chunk_size - 1) // chunk_size
                    chunk_checksum_list: List[str] = []

                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, num_slots)
                        chunk_data = kv_at_slots[start_idx:end_idx]
                        chunk_bytes = (
                            chunk_data.detach().cpu().contiguous().numpy().tobytes()
                        )
                        chunk_checksum = hashlib.md5(chunk_bytes).hexdigest()
                        chunk_checksum_list.append(chunk_checksum)

                    chunk_checksums[layer_name] = chunk_checksum_list

            result: Dict[str, Any] = {"layer_checksums": layer_checksums}
            if chunk_size is not None:
                result["chunk_checksums"] = chunk_checksums
                result["chunk_size"] = chunk_size
                result["num_chunks"] = (
                    (len(slot_indices) + chunk_size - 1) // chunk_size
                    if slot_indices
                    else 0
                )

            return result

        adapter.compute_kvcache_checksums = compute_checksums
        return adapter

    @pytest.fixture
    def client_with_adapter(self, mock_lmcache_adapter):
        """Create a test client with mocked adapter."""
        app.state.lmcache_adapter = mock_lmcache_adapter
        return TestClient(app)

    # ==========================================================================
    # Tests for /kvcache/check endpoint - Layer-level checksums
    # ==========================================================================

    def test_kvcache_check_success_layer_only(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test successful kvcache check with layer-level checksums only."""
        response = client_with_adapter.post("/kvcache/check?slot_mapping=0,1,2,3")

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["slot_mapping"] == [0, 1, 2, 3]
        assert "layer_checksums" in response_data
        assert "layer_0" in response_data["layer_checksums"]
        assert "layer_1" in response_data["layer_checksums"]
        # Verify checksums are valid MD5 hashes (32 hex chars)
        for checksum in response_data["layer_checksums"].values():
            assert len(checksum) == 32
            assert all(c in "0123456789abcdef" for c in checksum)

    def test_kvcache_check_missing_slot_mapping(self, client_with_adapter):
        """Test kvcache check with missing slot_mapping parameter."""
        response = client_with_adapter.post("/kvcache/check")

        assert response.status_code == 400
        response_data = json.loads(response.text)
        assert response_data["error"] == "Missing parameters"
        assert "slot_mapping" in response_data["message"]

    def test_kvcache_check_invalid_slot_mapping_format(self, client_with_adapter):
        """Test kvcache check with invalid slot_mapping format."""
        response = client_with_adapter.post("/kvcache/check?slot_mapping=a,b,c")

        assert response.status_code == 400
        response_data = json.loads(response.text)
        assert response_data["error"] == "Invalid slot_mapping format"

    def test_kvcache_check_no_adapter(self):
        """Test kvcache check when adapter is not available."""
        app.state.lmcache_adapter = None
        client = TestClient(app)

        response = client.post("/kvcache/check?slot_mapping=0,1,2,3")

        assert response.status_code == 503
        response_data = json.loads(response.text)
        assert response_data["error"] == "LMCache adapter unavailable"

    # ==========================================================================
    # Tests for /kvcache/check endpoint - Chunk-level checksums
    # ==========================================================================

    def test_kvcache_check_with_chunk_size(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test kvcache check with chunk_size for per-chunk checksums."""
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3,4,5,6,7&chunk_size=4"
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["slot_mapping"] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert response_data["chunk_size"] == 4
        assert response_data["num_chunks"] == 2

        # Verify layer checksums exist
        assert "layer_checksums" in response_data
        assert "layer_0" in response_data["layer_checksums"]
        assert "layer_1" in response_data["layer_checksums"]

        # Verify chunk checksums exist
        assert "chunk_checksums" in response_data
        assert "layer_0" in response_data["chunk_checksums"]
        assert "layer_1" in response_data["chunk_checksums"]
        assert len(response_data["chunk_checksums"]["layer_0"]) == 2
        assert len(response_data["chunk_checksums"]["layer_1"]) == 2

    def test_kvcache_check_chunk_size_uneven_division(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test chunk checksums when slots don't divide evenly by chunk_size."""
        # 7 slots with chunk_size=3 should give 3 chunks: [0-2], [3-5], [6]
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3,4,5,6&chunk_size=3"
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["num_chunks"] == 3
        assert len(response_data["chunk_checksums"]["layer_0"]) == 3

    def test_kvcache_check_chunk_size_equals_slots(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test when chunk_size equals the number of slots (single chunk)."""
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=4"
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["num_chunks"] == 1
        assert len(response_data["chunk_checksums"]["layer_0"]) == 1

        # The single chunk checksum should equal the layer checksum
        assert (
            response_data["chunk_checksums"]["layer_0"][0]
            == response_data["layer_checksums"]["layer_0"]
        )

    def test_kvcache_check_chunk_size_larger_than_slots(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test when chunk_size is larger than the number of slots."""
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=10"
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["num_chunks"] == 1
        assert len(response_data["chunk_checksums"]["layer_0"]) == 1

    def test_kvcache_check_invalid_chunk_size_zero(self, client_with_adapter):
        """Test kvcache check with invalid chunk_size=0."""
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=0"
        )

        assert response.status_code == 400
        response_data = json.loads(response.text)
        assert response_data["error"] == "Invalid chunk_size"

    def test_kvcache_check_invalid_chunk_size_negative(self, client_with_adapter):
        """Test kvcache check with negative chunk_size."""
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=-1"
        )

        assert response.status_code == 400
        response_data = json.loads(response.text)
        assert response_data["error"] == "Invalid chunk_size"

    def test_kvcache_check_chunk_size_one(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test with chunk_size=1 (each slot is its own chunk)."""
        response = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=1"
        )

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["num_chunks"] == 4
        assert len(response_data["chunk_checksums"]["layer_0"]) == 4

    # ==========================================================================
    # Tests for checksum consistency
    # ==========================================================================

    def test_kvcache_check_checksum_consistency(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test that checksums are consistent across multiple calls."""
        response1 = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        )
        response2 = client_with_adapter.post(
            "/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        )

        data1 = json.loads(response1.text)
        data2 = json.loads(response2.text)

        # Layer checksums should be identical
        assert data1["layer_checksums"] == data2["layer_checksums"]
        # Chunk checksums should be identical
        assert data1["chunk_checksums"] == data2["chunk_checksums"]

    def test_kvcache_check_different_slots_different_checksums(
        self, client_with_adapter, mock_lmcache_adapter
    ):
        """Test that different slots produce different checksums."""
        response1 = client_with_adapter.post("/kvcache/check?slot_mapping=0,1,2,3")
        response2 = client_with_adapter.post("/kvcache/check?slot_mapping=4,5,6,7")

        data1 = json.loads(response1.text)
        data2 = json.loads(response2.text)

        # Checksums should be different for different slots
        assert (
            data1["layer_checksums"]["layer_0"] != data2["layer_checksums"]["layer_0"]
        )

    # ==========================================================================
    # Tests for /kvcache/info endpoint
    # ==========================================================================

    def test_kvcache_info_success(self, client_with_adapter, mock_lmcache_adapter):
        """Test successful kvcache info retrieval."""
        response = client_with_adapter.get("/kvcache/info")

        assert response.status_code == 200
        response_data = json.loads(response.text)
        assert response_data["status"] == "success"
        assert response_data["num_layers"] == 2
        assert "layers" in response_data
        assert "layer_0" in response_data["layers"]
        assert "layer_1" in response_data["layers"]
        assert "shape" in response_data["layers"]["layer_0"]
        assert "dtype" in response_data["layers"]["layer_0"]
        assert "device" in response_data["layers"]["layer_0"]

    def test_kvcache_info_no_adapter(self):
        """Test kvcache info when adapter is not available."""
        app.state.lmcache_adapter = None
        client = TestClient(app)

        response = client.get("/kvcache/info")

        assert response.status_code == 503
        response_data = json.loads(response.text)
        assert response_data["error"] == "LMCache adapter unavailable"

    def test_kvcache_info_empty_kv_caches(self):
        """Test kvcache info when kv_caches is empty."""
        adapter = MagicMock()
        adapter.kv_caches = {}
        app.state.lmcache_adapter = adapter
        client = TestClient(app)

        response = client.get("/kvcache/info")

        assert response.status_code == 404
        response_data = json.loads(response.text)
        assert response_data["error"] == "kv_caches not available"

    # ==========================================================================
    # Tests for empty kv_caches
    # ==========================================================================

    def test_kvcache_check_empty_kv_caches(self):
        """Test kvcache check when kv_caches is empty."""
        adapter = MagicMock()
        adapter.kv_caches = {}
        adapter.compute_kvcache_checksums = MagicMock(return_value=None)
        app.state.lmcache_adapter = adapter
        client = TestClient(app)

        response = client.post("/kvcache/check?slot_mapping=0,1,2,3")

        assert response.status_code == 500
        response_data = json.loads(response.text)
        assert response_data["error"] == "Failed to compute checksums"
        assert "empty" in response_data["message"]
