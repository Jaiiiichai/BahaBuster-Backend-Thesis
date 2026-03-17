"""Health-related endpoints."""
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(request: Request) -> dict:
    """Return API and Supabase connectivity status."""

    supabase_status = getattr(
        request.app.state,
        "supabase_status",
        {"configured": False, "connected": False, "message": "Supabase not initialized."},
    )

    return {
        "status": "Backend is running fine tho",
        "supabase": supabase_status,
    }
