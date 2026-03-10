"""Prediction-facing API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..dependencies import get_prediction_service
from ..services.prediction_service import PredictionService

router = APIRouter(tags=["predictions"])


@router.get("/predict_flood")
def predict_flood(barangay: str, service: PredictionService = Depends(get_prediction_service)):
    try:
        payload = service.predict_flood(barangay)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(payload)
