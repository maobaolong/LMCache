# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path

# Third Party
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "static"


@router.get("/")
async def root():
    """Serve the frontend dashboard if available,
    otherwise return a basic liveness check."""
    index = _FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(str(index))
    return {"status": "ok", "service": "LMCache HTTP API"}
