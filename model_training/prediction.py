"""Prediction helpers that wrap trained models with weather and manual inputs."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
import requests

from .config import CLASSIFICATION_THRESHOLD
from .features import create_features
from .registry import get_model_for_barangay


def risk_level(probability: float) -> str:
    """Convert a flood probability into the categorical risk buckets used downstream."""

    if probability < 0.1:
        return "LOW"
    if probability < 0.3:
        return "MODERATE"
    if probability < 0.6:
        return "HIGH"
    return "SEVERE"


def format_prediction_summary(prob: float, depth: float, level: str) -> str:
    """Return a single formatted string summarizing key prediction outputs."""

    return f"Flood probability {prob*100:.1f}%, risk level {level}, expected depth {depth:.1f} cm."


def get_weather() -> pd.DataFrame:
    """Fetch a short-term weather forecast and align the schema with training data."""

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.3157,
        "longitude": 123.8854,
        "daily": "temperature_2m_max,relative_humidity_2m_max,precipitation_sum",
        "timezone": "auto",
    }
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    data = res.json()["daily"]

    df = pd.DataFrame(
        {
            "temperature_2m (°C)": data["temperature_2m_max"],
            "relative_humidity_2m (%)": data["relative_humidity_2m_max"],
            "rain": data["precipitation_sum"],
        }
    )
    df["precipitation"] = df["rain"]
    df = create_features(df)
    return df.head(3)


def run_manual_prediction(feature_map: Dict[str, float], model_bundle: dict) -> Tuple[Dict[str, float], float, float, int]:
    """Apply manual feature inputs to the trained models and return the prediction tuple."""

    features = model_bundle["features"]
    rf_model = model_bundle["rf"]
    xgb_model = model_bundle["xgb"]
    reg_model = model_bundle.get("reg")

    row = {feature: float(feature_map.get(feature, 0.0)) for feature in features}
    X = pd.DataFrame([row])

    rf_prob = rf_model.predict_proba(X)[0][1]
    xgb_prob = xgb_model.predict_proba(X)[0][1]
    prob = (rf_prob + xgb_prob) / 2

    flood = int(prob > CLASSIFICATION_THRESHOLD)
    depth = float(reg_model.predict(X)[0]) if reg_model is not None and flood == 1 else 0.0

    return row, float(prob), depth, flood


def predict_with_weather(barangay: str, registry: Optional[dict[str, dict]] = None) -> dict:
    """Generate short-term forecasts for a barangay using freshly pulled weather data."""

    model_bundle = get_model_for_barangay(barangay, registry)
    features = model_bundle["features"]
    rf_model = model_bundle["rf"]
    xgb_model = model_bundle["xgb"]
    reg_model = model_bundle.get("reg")
    barangay_metrics = model_bundle["metrics"]

    weather = get_weather()
    results = []

    for i, row in weather.iterrows():
        valid_features = [f for f in features if f in row.index]
        X = row[valid_features].to_frame().T

        for feature in features:
            if feature not in X.columns:
                X[feature] = 0

        rf_prob = rf_model.predict_proba(X[features])[0][1]
        xgb_prob = xgb_model.predict_proba(X[features])[0][1]
        prob = (rf_prob + xgb_prob) / 2
        flood = int(prob > CLASSIFICATION_THRESHOLD)
        depth = float(reg_model.predict(X[features])[0]) if reg_model is not None and flood == 1 else 0.0
        rl = risk_level(prob)

        results.append(
            {
                "day": i + 1,
                "flood_probability": round(float(prob), 4),
                "predicted_depth_cm": round(float(depth), 2),
                "risk_level": rl,
                "alert": flood,
                "summary": format_prediction_summary(prob, depth, rl),
            }
        )

    return {"barangay": model_bundle["barangay"], "predictions": results, "metrics": barangay_metrics}


def manual_prediction_response(
    barangay: str,
    features: Dict[str, float],
    actual_flood: Optional[bool] = None,
    actual_depth_cm: Optional[float] = None,
    registry: Optional[dict[str, dict]] = None,
) -> dict:
    """Run a manual prediction and attach optional ground-truth annotations."""

    if not features:
        raise ValueError("'features' must contain at least one numeric value.")

    model_bundle = get_model_for_barangay(barangay, registry)
    feature_row, prob, depth, flood = run_manual_prediction(features, model_bundle)

    actuals = None
    if actual_flood is not None or actual_depth_cm is not None:
        actuals = {
            "flood_occurred": actual_flood,
            "depth_cm": actual_depth_cm,
            "is_alert_correct": None if actual_flood is None else (bool(flood) == bool(actual_flood)),
        }

        if actual_depth_cm is not None:
            actuals["depth_error_cm"] = None if depth is None else round(actual_depth_cm - depth, 2)

    rl = risk_level(prob)
    prediction = {
        "flood_probability": round(prob, 4),
        "predicted_depth_cm": round(depth, 2),
        "risk_level": rl,
        "alert": flood,
        "summary": format_prediction_summary(prob, depth, rl),
    }

    return {
        "barangay": model_bundle["barangay"],
        "prediction": prediction,
        "input_features": feature_row,
        "actuals": actuals,
        "metrics": model_bundle["metrics"],
    }
