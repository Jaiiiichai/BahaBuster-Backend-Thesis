"""Health-related endpoints."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple status payload indicating the API is live."""

    return {"status": "Backend is running fine tho"}
