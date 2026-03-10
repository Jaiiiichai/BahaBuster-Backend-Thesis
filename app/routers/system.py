"""System-level endpoints such as health checks."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/test")
def test_endpoint():
    return {"message": "This is a test endpoint."}


@router.get("/health")
def healthcheck():
    return {"status": "ok"}
