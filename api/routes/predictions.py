"""Prediction endpoints that expose the model-training package over HTTP."""
from fastapi import APIRouter, HTTPException, Query

from model_training import (
    ModelNotFoundError,
    get_available_barangays,
    manual_prediction_response,
    predict_with_weather,
)

from ..schemas import ManualPredictionRequest

router = APIRouter(tags=["predictions"])


@router.get("/predict/{barangay}")
def predict(barangay: str):
    """Return the latest weather-driven forecast for the provided barangay."""

    try:
        return predict_with_weather(barangay)
    except ModelNotFoundError as exc:
        available = ", ".join(exc.available) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{barangay}'. Available: {available}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/predict_flood")
def predict_flood(barangay: str = Query(..., description="Barangay name")):
    """Convenience endpoint that mirrors predict() but uses a query string parameter."""

    return predict(barangay)


@router.get("/predict_all")
def predict_all_barangays():
    """Return weather-driven predictions for every barangay with a trained model."""

    barangays = get_available_barangays()
    if not barangays:
        raise HTTPException(status_code=404, detail="No available barangay models found.")

    predictions = []
    failed = []

    for barangay in barangays:
        try:
            predictions.append(predict_with_weather(barangay))
        except Exception as exc:
            failed.append({"barangay": barangay, "error": str(exc)})

    return {
        "count": len(predictions),
        "barangays": predictions,
        "failed": failed,
    }


@router.post("/predict_manual")
def predict_manual(payload: ManualPredictionRequest):
    """Execute a manual prediction using explicitly provided features."""

    if not payload.barangay:
        raise HTTPException(status_code=400, detail="'barangay' is required for manual predictions.")

    try:
        return manual_prediction_response(
            barangay=payload.barangay,
            features=payload.features,
            actual_flood=payload.actual_flood,
            actual_depth_cm=payload.actual_depth_cm,
        )
    except ModelNotFoundError as exc:
        available = ", ".join(exc.available) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{payload.barangay}'. Available: {available}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
