"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 when the app process is up."""
    return {"status": "ok", "version": __version__}
