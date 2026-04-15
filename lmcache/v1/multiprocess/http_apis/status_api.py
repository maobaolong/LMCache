# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any
import json

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


class IndentedJSONResponse(JSONResponse):
    """JSONResponse with indented output for readability."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")


@router.get("/api/status")
@router.get("/inference_info")
async def status(request: Request) -> Any:
    """
    Detailed status endpoint for inspecting internal state
    of all MP components (L1 cache, L2 adapters, controllers,
    sessions).
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )
    return IndentedJSONResponse(
        content=engine.report_status(),
    )
