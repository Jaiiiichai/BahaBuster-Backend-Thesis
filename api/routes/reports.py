"""Flood report endpoints backed by Supabase."""
from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import FloodReportCreateRequest, FloodReportResponse
from ..supabase_client import fetch_flood_reports_by_barangay, insert_flood_report

router = APIRouter(tags=["reports"])


@router.post("/reports", response_model=FloodReportResponse, status_code=201)
def create_report(payload: FloodReportCreateRequest, request: Request):
    """Create a flood report in Supabase."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        report = insert_flood_report(supabase_client, payload.model_dump(mode="json"))
        return report
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/reports", response_model=list[FloodReportResponse])
def get_reports_by_barangay(
    request: Request,
    barangay: str = Query(..., description="Barangay name used to filter reports case-insensitively"),
):
    """Return flood reports filtered by barangay name case-insensitively."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return fetch_flood_reports_by_barangay(supabase_client, barangay)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
