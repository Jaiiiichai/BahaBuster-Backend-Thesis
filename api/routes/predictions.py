"""Prediction endpoints that expose the model-training package over HTTP."""
from fastapi import APIRouter, HTTPException, Query

from model_training import (
    ModelNotFoundError,
    get_available_barangays,
    get_weather,
    manual_prediction_response,
    predict_with_moderate_weather_target,
    predict_with_target_risk_level,
    predict_with_weather_depth_diagnostics,
    predict_with_weather,
    find_shared_intensity,
    predict_with_shared_intensity,
    predict_all_shared_intensity,
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


@router.get("/predict_depth_diagnostics/{barangay}")
def predict_depth_diagnostics(barangay: str):
    """Testing endpoint returning expected and conditional depth diagnostics."""

    try:
        return predict_with_weather_depth_diagnostics(barangay)
    except ModelNotFoundError as exc:
        available = ", ".join(exc.available) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{barangay}'. Available: {available}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/predict_all")
def predict_all_barangays():
    """Return weather-driven predictions for every barangay with a trained model."""

    barangays = get_available_barangays()
    if not barangays:
        raise HTTPException(status_code=404, detail="No available barangay models found.")

    predictions = []
    failed = []
    weather = get_weather()

    for barangay in barangays:
        try:
            predictions.append(predict_with_weather(barangay, weather=weather))
        except Exception as exc:
            failed.append({"barangay": barangay, "error": str(exc)})

    return {
        "count": len(predictions),
        "barangays": predictions,
        "failed": failed,
    }


@router.get("/predict_all_depth_diagnostics")
def predict_all_depth_diagnostics():
    """Testing endpoint returning depth diagnostics for every barangay."""

    barangays = get_available_barangays()
    if not barangays:
        raise HTTPException(status_code=404, detail="No available barangay models found.")

    predictions = []
    failed = []
    weather = get_weather()

    for barangay in barangays:
        try:
            predictions.append(predict_with_weather_depth_diagnostics(barangay, weather=weather))
        except Exception as exc:
            failed.append({"barangay": barangay, "error": str(exc)})

    return {
        "count": len(predictions),
        "barangays": predictions,
        "failed": failed,
    }


@router.get("/predict_all_moderate")
def predict_all_barangays_moderate(
    target_probability: float = Query(0.55, ge=0.05, le=0.95, description="Desired day-1 flood probability."),
    tolerance: float = Query(0.1, ge=0.01, le=0.4, description="Acceptable distance from target probability."),
):
    """Return all-barangay predictions under a moderate weather stress scenario."""

    target_probability = float(target_probability)
    tolerance = float(tolerance)

    barangays = get_available_barangays()
    if not barangays:
        raise HTTPException(status_code=404, detail="No available barangay models found.")

    predictions = []
    failed = []
    weather = get_weather()

    for barangay in barangays:
        try:
            predictions.append(
                predict_with_moderate_weather_target(
                    barangay=barangay,
                    target_probability=target_probability,
                    tolerance=tolerance,
                    weather=weather,
                )
            )
        except Exception as exc:
            failed.append({"barangay": barangay, "error": str(exc)})

    within_band_count = sum(1 for item in predictions if item.get("within_target_band"))

    return {
        "count": len(predictions),
        "target_probability": round(target_probability, 2),
        "tolerance": round(tolerance, 2),
        "within_target_band_count": within_band_count,
        "barangays": predictions,
        "failed": failed,
    }


@router.get("/predict_all_test_moderate_risk")
def predict_all_barangays_test_moderate_risk():
    """Return all-barangay test predictions using a single shared intensity targeting MODERATE risk."""

    barangays = get_available_barangays()
    if not barangays:
        raise HTTPException(status_code=404, detail="No available barangay models found.")

    weather = get_weather()
    result = predict_all_shared_intensity(barangays, target_risk_level="MODERATE", weather=weather)
    achieved_count = sum(1 for item in result["barangays"] if item.get("achieved_target_risk"))

    return {
        "count": len(result["barangays"]),
        "target_risk_level": "MODERATE",
        "shared_intensity": result["shared_intensity"],
        "achieved_target_risk_count": achieved_count,
        "barangays": result["barangays"],
        "failed": result["failed"],
    }


@router.get("/predict_all_test_severe_risk")
def predict_all_barangays_test_severe_risk():
    """Return all-barangay test predictions using a single shared intensity targeting SEVERE risk."""

    barangays = get_available_barangays()
    if not barangays:
        raise HTTPException(status_code=404, detail="No available barangay models found.")

    weather = get_weather()
    result = predict_all_shared_intensity(barangays, target_risk_level="SEVERE", weather=weather)
    achieved_count = sum(1 for item in result["barangays"] if item.get("achieved_target_risk"))

    return {
        "count": len(result["barangays"]),
        "target_risk_level": "SEVERE",
        "shared_intensity": result["shared_intensity"],
        "achieved_target_risk_count": achieved_count,
        "barangays": result["barangays"],
        "failed": result["failed"],
    }


@router.get("/predict_test_moderate_risk/{barangay}")
def predict_barangay_test_moderate_risk(barangay: str):
    """Return a single-barangay test prediction using the shared MODERATE risk intensity."""

    weather = get_weather()
    try:
        result = predict_all_shared_intensity([barangay], target_risk_level="MODERATE", weather=weather)
    except ModelNotFoundError as exc:
        available = ", ".join(exc.available) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{barangay}'. Available: {available}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["failed"]:
        raise HTTPException(status_code=500, detail=result["failed"][0]["error"])

    barangay_result = result["barangays"][0]
    return {
        "barangay": barangay_result["barangay"],
        "target_risk_level": "MODERATE",
        "shared_intensity": result["shared_intensity"],
        "achieved_target_risk": barangay_result["achieved_target_risk"],
        "predictions": barangay_result["predictions"],
    }


@router.get("/predict_test_severe_risk/{barangay}")
def predict_barangay_test_severe_risk(barangay: str):
    """Return a single-barangay test prediction using the shared SEVERE risk intensity."""

    weather = get_weather()
    try:
        result = predict_all_shared_intensity([barangay], target_risk_level="SEVERE", weather=weather)
    except ModelNotFoundError as exc:
        available = ", ".join(exc.available) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{barangay}'. Available: {available}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["failed"]:
        raise HTTPException(status_code=500, detail=result["failed"][0]["error"])

    barangay_result = result["barangays"][0]
    return {
        "barangay": barangay_result["barangay"],
        "target_risk_level": "SEVERE",
        "shared_intensity": result["shared_intensity"],
        "achieved_target_risk": barangay_result["achieved_target_risk"],
        "predictions": barangay_result["predictions"],
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
