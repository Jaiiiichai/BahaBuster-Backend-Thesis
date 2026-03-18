"""Alert endpoints backed by Supabase."""
from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import AlertCreateRequest, AlertResponse
from ..supabase_client import fetch_alerts_by_barangay, insert_alert

router = APIRouter(tags=["alerts"])


@router.post("/alerts", response_model=AlertResponse, status_code=201)
def create_alert(payload: AlertCreateRequest, request: Request):
    """Create an alert in Supabase."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return insert_alert(supabase_client, payload.model_dump(mode="json"))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/alerts", response_model=list[AlertResponse])
def get_alerts_by_barangay(
    request: Request,
    barangay: str = Query(..., description="Barangay name used to filter alerts case-insensitively"),
):
    """Return alerts filtered by barangay/location case-insensitively."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return fetch_alerts_by_barangay(supabase_client, barangay)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
