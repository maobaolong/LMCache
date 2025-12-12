# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import hashlib
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


def _create_error_response(error_info: dict, status_code: int) -> PlainTextResponse:
    """Create a standardized error response."""
    return PlainTextResponse(
        content=json.dumps(error_info, indent=2),
        media_type="application/json",
        status_code=status_code,
    )


def _compute_tensor_checksum(tensor: torch.Tensor) -> str:
    """Compute MD5 checksum of a tensor."""
    # Move to CPU and convert to bytes for hashing
    tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()


@router.post("/kvcache/check")
async def kvcache_check(
    request: Request,
    slot_mapping: Optional[str] = None,
    chunk_size: Optional[int] = None,
):
    """Compute checksum for kvcaches at specified slot_mapping positions.

    This endpoint is used to verify that stored and retrieved kvcaches are identical.

    Args:
        request (Request): The FastAPI request object containing application state.
        slot_mapping (Optional[str], optional): Comma-separated list of slot indices.
            Example: "0,1,2,3" or "0,1,2,3,4,5,6,7". Defaults to None.
        chunk_size (Optional[int], optional): If provided, also compute per-chunk
            checksums. Each chunk contains `chunk_size` slots.

    Returns:
        PlainTextResponse: A JSON response containing checksums for each layer.

    Example:
        ```bash
        # Without chunk_size (layer-level checksums only)
        curl -X POST "http://localhost:8000/kvcache/check?slot_mapping=0,1,2,3"

        # With chunk_size (both layer-level and chunk-level checksums)
        curl -X POST "http://localhost:8000/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        # Response: {
        #   "status": "success",
        #   "slot_mapping": [0, 1, 2, 3],
        #   "chunk_size": 2,
        #   "num_chunks": 2,
        #   "layer_checksums": {
        #       "layer_0": "abc123...",
        #       "layer_1": "def456...",
        #   },
        #   "chunk_checksums": {
        #       "layer_0": ["checksum_chunk0", "checksum_chunk1"],
        #       "layer_1": ["checksum_chunk0", "checksum_chunk1"],
        #   }
        # }
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        if not lmcache_adapter:
            return _create_error_response(
                {
                    "error": "LMCache adapter unavailable",
                    "message": "LMCache adapter not configured.",
                },
                503,
            )

        if not slot_mapping:
            return _create_error_response(
                {
                    "error": "Missing parameters",
                    "message": "slot_mapping parameter is required",
                },
                400,
            )

        # Parse slot_mapping from comma-separated string
        try:
            slot_indices = [int(s.strip()) for s in slot_mapping.split(",")]
        except ValueError as e:
            return _create_error_response(
                {
                    "error": "Invalid slot_mapping format",
                    "message": "slot_mapping must be comma-separated integers: %s"
                    % str(e),
                },
                400,
            )

        # Validate chunk_size if provided
        if chunk_size is not None and chunk_size <= 0:
            return _create_error_response(
                {
                    "error": "Invalid chunk_size",
                    "message": "chunk_size must be a positive integer",
                },
                400,
            )

        # Get checksums from the adapter
        checksums_result = lmcache_adapter.compute_kvcache_checksums(
            slot_indices, chunk_size
        )

        if checksums_result is None:
            return _create_error_response(
                {
                    "error": "Failed to compute checksums",
                    "message": "kv_caches not available or empty",
                },
                500,
            )

        response_data = {
            "status": "success",
            "slot_mapping": slot_indices,
            "layer_checksums": checksums_result.get("layer_checksums", {}),
        }

        # Include chunk-level checksums if chunk_size was provided
        if chunk_size is not None:
            response_data["chunk_size"] = checksums_result.get("chunk_size")
            response_data["num_chunks"] = checksums_result.get("num_chunks")
            response_data["chunk_checksums"] = checksums_result.get(
                "chunk_checksums", {}
            )

        return PlainTextResponse(
            content=json.dumps(response_data, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("Failed to compute kvcache checksums: %s", str(e))
        return _create_error_response(
            {"error": "Failed to compute checksums", "message": str(e)},
            500,
        )


@router.get("/kvcache/info")
async def kvcache_info(request: Request):
    """Get information about the current kvcaches.

    Returns information about the kvcaches structure including layer names,
    shapes, and device information.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A JSON response containing kvcache information.
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        if not lmcache_adapter:
            return _create_error_response(
                {
                    "error": "LMCache adapter unavailable",
                    "message": "LMCache adapter not configured.",
                },
                503,
            )

        kv_caches = getattr(lmcache_adapter, "kv_caches", None)
        if not kv_caches:
            return _create_error_response(
                {
                    "error": "kv_caches not available",
                    "message": "kv_caches is empty or not initialized",
                },
                404,
            )

        layers_info: dict = {}
        for layer_name, kv_tensor in kv_caches.items():
            layers_info[layer_name] = {
                "shape": list(kv_tensor.shape),
                "dtype": str(kv_tensor.dtype),
                "device": str(kv_tensor.device),
            }

        info = {
            "status": "success",
            "num_layers": len(kv_caches),
            "layers": layers_info,
        }

        return PlainTextResponse(
            content=json.dumps(info, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("Failed to get kvcache info: %s", str(e))
        return _create_error_response(
            {"error": "Failed to get kvcache info", "message": str(e)},
            500,
        )
