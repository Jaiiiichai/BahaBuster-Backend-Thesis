"""Prediction helpers that wrap trained models with weather and manual inputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

from .config import CLASSIFICATION_THRESHOLD, TARGET_CLASS
from .data_loader import load_training_dataframe
from .features import create_features
from .registry import get_model_for_barangay


def risk_level(depth_cm: float) -> str:
    """Convert predicted flood depth (cm) into categorical risk buckets."""

    if depth_cm < 7:
        return "LOW"
    if depth_cm < 15:
        return "MODERATE"
    return "SEVERE"


def display_depth_cm(depth_cm: float) -> int:
    """Return a user-friendly whole-centimeter depth value."""

    return max(0, int(round(float(depth_cm))))


def format_prediction_summary(prob: float, depth: float, level: str) -> str:
    """Return a single formatted string summarizing key prediction outputs."""

    return f"Flood probability {prob*100:.1f}%, risk level {level}, expected depth {display_depth_cm(depth)} cm."


@lru_cache(maxsize=1)
def _load_feature_defaults() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Build cached numeric median defaults for each barangay and globally."""

    df = create_features(load_training_dataframe().drop_duplicates().reset_index(drop=True))
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    global_defaults = {
        key: float(value)
        for key, value in numeric_df.median(numeric_only=True).to_dict().items()
        if pd.notna(value)
    }

    barangay_defaults: dict[str, dict[str, float]] = {}
    if "barangay" in df.columns:
        for barangay_name, group in df.groupby("barangay"):
            group_numeric = group.apply(pd.to_numeric, errors="coerce")
            medians = {
                key: float(value)
                for key, value in group_numeric.median(numeric_only=True).to_dict().items()
                if pd.notna(value)
            }
            barangay_defaults[barangay_name] = {**global_defaults, **medians}

    return barangay_defaults, global_defaults


def _feature_defaults_for_barangay(barangay: str) -> dict[str, float]:
    """Return barangay-specific feature defaults, falling back to global medians."""

    barangay_defaults, global_defaults = _load_feature_defaults()
    return barangay_defaults.get(barangay, global_defaults)


@lru_cache(maxsize=1)
def _load_flood_condition_profiles() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Build numeric median profiles from rows where flooding occurred."""

    df = create_features(load_training_dataframe().drop_duplicates().reset_index(drop=True))
    if TARGET_CLASS not in df.columns:
        return {}, {}

    flood_df = df[pd.to_numeric(df[TARGET_CLASS], errors="coerce") >= 1].copy()
    if flood_df.empty:
        return {}, {}

    global_profile = {
        key: float(value)
        for key, value in flood_df.apply(pd.to_numeric, errors="coerce").median(numeric_only=True).to_dict().items()
        if pd.notna(value)
    }

    barangay_profiles: dict[str, dict[str, float]] = {}
    if "barangay" in flood_df.columns:
        for barangay_name, group in flood_df.groupby("barangay"):
            medians = {
                key: float(value)
                for key, value in group.apply(pd.to_numeric, errors="coerce").median(numeric_only=True).to_dict().items()
                if pd.notna(value)
            }
            barangay_profiles[barangay_name] = {**global_profile, **medians}

    return barangay_profiles, global_profile


def _flood_profile_for_barangay(barangay: str) -> dict[str, float]:
    """Return flood-condition profile for the barangay, falling back globally."""

    profiles, global_profile = _load_flood_condition_profiles()
    return profiles.get(barangay, global_profile)


def _build_feature_row(row: pd.Series, features: list[str], barangay: str) -> pd.DataFrame:
    """Create a complete model input row using weather values and robust fallback defaults."""

    defaults = _feature_defaults_for_barangay(barangay)
    values: dict[str, float] = {}

    for feature in features:
        raw_value = row.get(feature, None)
        if raw_value is None or pd.isna(raw_value):
            values[feature] = float(defaults.get(feature, 0.0))
        else:
            values[feature] = float(raw_value)

    return pd.DataFrame([values])


def get_weather() -> pd.DataFrame:
    """Fetch a short-term weather forecast and align the schema with training data."""

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.3157,
        "longitude": 123.8854,
        "daily": "temperature_2m_max,relative_humidity_2m_max,precipitation_sum,wind_speed_10m_max,surface_pressure_mean",
        "hourly": "precipitation",
        "wind_speed_unit": "ms",
        "timezone": "auto",
    }
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    payload = res.json()
    data = payload["daily"]
    hourly = payload.get("hourly", {})
    hourly_precip = hourly.get("precipitation", [])

    rain_3hr_total = []
    rain_6hr_total = []
    rain_24hr_total = []
    day_count = len(data["precipitation_sum"])

    for day_idx in range(day_count):
        start = day_idx * 24
        chunk = hourly_precip[start : start + 24]
        if not chunk:
            rain_3hr_total.append(0.0)
            rain_6hr_total.append(0.0)
            rain_24hr_total.append(0.0)
            continue

        rain_3hr_total.append(float(sum(chunk[-3:])))
        rain_6hr_total.append(float(sum(chunk[-6:])))
        rain_24hr_total.append(float(sum(chunk)))

    df = pd.DataFrame(
        {
            "temperature_2m (°C)": data["temperature_2m_max"],
            "relative_humidity_2m (%)": data["relative_humidity_2m_max"],
            "rain": data["precipitation_sum"],
            "rain_3hr_total": rain_3hr_total,
            "rain_6hr_total": rain_6hr_total,
            "rain_24hr_total": rain_24hr_total,
            "wind_speed (m/s)": data["wind_speed_10m_max"],
            "atmospheric_pressure (hPa)": data["surface_pressure_mean"],
        }
    )
    df["precipitation"] = df["rain"]
    df = create_features(df)
    return df.head(3)


def build_moderate_weather_scenario(weather: pd.DataFrame, intensity: float, barangay: Optional[str] = None) -> pd.DataFrame:
    """Return a weather frame adjusted by an intensity factor for scenario testing."""

    scenario = weather.copy()
    rain_multiplier = float(intensity) ** 2
    blend = min(1.0, max(0.0, (float(intensity) - 1.0) / 4.0))
    flood_profile = _flood_profile_for_barangay(barangay) if barangay else {}

    profile_columns = [
        "temperature_2m (°C)",
        "relative_humidity_2m (%)",
        "rain",
        "precipitation",
        "rain_3hr_total",
        "rain_6hr_total",
        "rain_24hr_total",
        "wind_speed (m/s)",
        "atmospheric_pressure (hPa)",
        "flood_frequency",
        "elevation (m)",
        "slope (degrees)",
    ]

    for col in profile_columns:
        if col not in flood_profile:
            continue

        target = float(flood_profile[col])
        if col in scenario.columns:
            base = pd.to_numeric(scenario[col], errors="coerce").fillna(target)
            scenario[col] = base * (1.0 - blend) + target * blend
        else:
            scenario[col] = target

    rain_columns = ["rain", "precipitation", "rain_3hr_total", "rain_6hr_total", "rain_24hr_total"]
    for col in rain_columns:
        if col in scenario.columns:
            scenario[col] = pd.to_numeric(scenario[col], errors="coerce").fillna(0.0) * rain_multiplier

    if "relative_humidity_2m (%)" in scenario.columns:
        humidity = pd.to_numeric(scenario["relative_humidity_2m (%)"], errors="coerce").fillna(0.0)
        scenario["relative_humidity_2m (%)"] = (humidity + (intensity - 1.0) * 4.0).clip(upper=98.0)

    if "wind_speed (m/s)" in scenario.columns:
        wind = pd.to_numeric(scenario["wind_speed (m/s)"], errors="coerce").fillna(0.0)
        scenario["wind_speed (m/s)"] = wind * (1.0 + (intensity - 1.0) * 0.5)

    if "atmospheric_pressure (hPa)" in scenario.columns:
        pressure = pd.to_numeric(scenario["atmospheric_pressure (hPa)"], errors="coerce").fillna(1012.0)
        scenario["atmospheric_pressure (hPa)"] = pressure - (intensity - 1.0) * 5.0

    if "temperature_2m (°C)" in scenario.columns:
        temp = pd.to_numeric(scenario["temperature_2m (°C)"], errors="coerce").fillna(27.0)
        scenario["temperature_2m (°C)"] = temp - (intensity - 1.0) * 0.3

    # Synthetic risk-tilt features used by some trained models.
    scenario["flood_frequency"] = 50.0 + (intensity - 1.0) * 20.0
    scenario["elevation (m)"] = max(1.0, 8.0 - (intensity - 1.0) * 0.6)
    scenario["slope (degrees)"] = max(0.5, 2.5 - (intensity - 1.0) * 0.2)

    return scenario


def predict_with_moderate_weather_target(
    barangay: str,
    target_probability: float = 0.55,
    tolerance: float = 0.1,
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
) -> dict:
    """Tune scenario intensity so day-1 flood probability approaches the target range."""

    if not (0.05 <= target_probability <= 0.95):
        raise ValueError("target_probability must be between 0.05 and 0.95")

    if not (0.01 <= tolerance <= 0.4):
        raise ValueError("tolerance must be between 0.01 and 0.4")

    base_weather = weather if weather is not None else get_weather()

    best_result: Optional[dict] = None
    best_distance = float("inf")
    best_intensity = 1.0

    intensity_grid = [1.1, 1.4, 1.8, 2.2, 2.8, 3.4, 4.2, 5.2, 6.5]
    for intensity in intensity_grid:
        scenario_weather = build_moderate_weather_scenario(base_weather, intensity, barangay=barangay)
        prediction = predict_with_weather(barangay, registry=registry, weather=scenario_weather)
        day1_probability = float(prediction["predictions"][0]["flood_probability"])
        distance = abs(day1_probability - target_probability)

        if distance < best_distance:
            best_distance = distance
            best_result = prediction
            best_intensity = intensity

        if distance <= tolerance:
            best_result = prediction
            best_intensity = intensity
            break

    if best_result is None:
        raise ValueError(f"Unable to generate moderate-weather prediction for {barangay}.")

    return {
        "barangay": best_result["barangay"],
        "target_probability": round(target_probability, 2),
        "selected_intensity": round(best_intensity, 2),
        "day1_probability": best_result["predictions"][0]["flood_probability"],
        "within_target_band": best_distance <= tolerance,
        "distance_to_target": round(best_distance, 4),
        "predictions": best_result["predictions"],
        "metrics": best_result["metrics"],
    }


def _distance_to_target_risk_band(depth_cm: float, target_risk_level: str) -> float:
    """Return distance in centimeters from the requested risk-level depth band."""

    level = target_risk_level.upper()
    if level == "MODERATE":
        if depth_cm < 7:
            return 7.0 - depth_cm
        if depth_cm < 15:
            return 0.0
        return depth_cm - 15.0

    if level == "SEVERE":
        return 0.0 if depth_cm >= 15 else (15.0 - depth_cm)

    raise ValueError(f"Unsupported target risk level: {target_risk_level}")


# Empirically determined intensities that produce representative MODERATE/SEVERE results
# across all barangays. Avoids a slow grid search at runtime.
_DEFAULT_INTENSITY: dict[str, float] = {
    "MODERATE": 2.4,
    "SEVERE": 2.8,
}


def predict_all_shared_intensity(
    barangays: list,
    target_risk_level: str = "MODERATE",
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
    intensity: Optional[float] = None,
) -> dict:
    """Run scenario predictions for all barangays at a single shared intensity.

    Uses a fixed empirical intensity by default (no grid search) so the response
    is fast: just 9 × 3-day predictions.  Pass ``intensity`` to override.
    Returns {"shared_intensity": float, "barangays": [per-barangay dicts], "failed": [...]}.
    """

    normalized_target = target_risk_level.strip().upper()
    if normalized_target not in {"MODERATE", "SEVERE"}:
        raise ValueError("target_risk_level must be one of: MODERATE, SEVERE")

    base_weather = weather if weather is not None else get_weather()
    chosen_intensity = float(intensity) if intensity is not None else _DEFAULT_INTENSITY[normalized_target]

    def _run_barangay(barangay: str) -> dict:
        try:
            scenario_weather = build_moderate_weather_scenario(base_weather, chosen_intensity, barangay=barangay)
            prediction = predict_with_weather(barangay, registry=registry, weather=scenario_weather)
            day1 = prediction["predictions"][0]
            depth_cm = float(day1["predicted_depth_cm"])
            dist = _distance_to_target_risk_band(depth_cm, normalized_target)
            return {
                "barangay": prediction["barangay"],
                "target_risk_level": normalized_target,
                "shared_intensity": round(chosen_intensity, 2),
                "achieved_target_risk": str(day1["risk_level"]).upper() == normalized_target,
                "distance_to_target_cm": round(dist, 2),
                "predictions": prediction["predictions"],
                "metrics": prediction["metrics"],
            }
        except Exception as exc:
            return {"barangay": barangay, "error": str(exc)}

    barangay_results: list = []
    for barangay in barangays:
        barangay_results.append(_run_barangay(barangay))

    return {
        "shared_intensity": round(chosen_intensity, 2),
        "barangays": barangay_results,
        "failed": [r for r in barangay_results if "error" in r],
    }


# Keep these for backwards compatibility / single-barangay use
def find_shared_intensity(
    barangays: list,
    target_risk_level: str = "MODERATE",
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
) -> float:
    """Deprecated helper — prefer predict_all_shared_intensity for efficiency."""
    result = predict_all_shared_intensity(barangays, target_risk_level, registry, weather)
    return result["shared_intensity"]


def predict_with_shared_intensity(
    barangay: str,
    intensity: float,
    target_risk_level: str = "MODERATE",
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
) -> dict:
    """Run a scenario prediction for a single barangay at a fixed intensity."""

    normalized_target = target_risk_level.strip().upper()
    base_weather = weather if weather is not None else get_weather()
    scenario_weather = build_moderate_weather_scenario(base_weather, intensity, barangay=barangay)
    prediction = predict_with_weather(barangay, registry=registry, weather=scenario_weather)
    day1 = prediction["predictions"][0]
    depth_cm = float(day1["predicted_depth_cm"])
    return {
        "barangay": prediction["barangay"],
        "target_risk_level": normalized_target,
        "shared_intensity": round(intensity, 2),
        "achieved_target_risk": str(day1["risk_level"]).upper() == normalized_target,
        "distance_to_target_cm": round(_distance_to_target_risk_band(depth_cm, normalized_target), 2),
        "predictions": prediction["predictions"],
        "metrics": prediction["metrics"],
    }


def predict_with_target_risk_level(
    barangay: str,
    target_risk_level: str = "MODERATE",
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
) -> dict:
    """Search scenario intensities for a day-1 prediction that reaches the target risk level."""

    normalized_target = target_risk_level.strip().upper()
    if normalized_target not in {"MODERATE", "SEVERE"}:
        raise ValueError("target_risk_level must be one of: MODERATE, SEVERE")

    base_weather = weather if weather is not None else get_weather()
    # Fine-grained steps in the low range to catch the narrow MODERATE band (7-15 cm);
    # coarser steps at higher intensities are sufficient for SEVERE (≥15 cm).
    if normalized_target == "MODERATE":
        intensity_grid = [
            1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
            1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.9, 2.0, 2.1, 2.2,
            2.4, 2.6, 2.8, 3.2, 3.8, 4.5,
        ]
    else:
        intensity_grid = [1.1, 1.3, 1.5, 1.8, 2.2, 2.8, 3.4, 4.2, 5.2, 6.5, 8.0, 10.0]

    best_result: Optional[dict] = None
    best_intensity = intensity_grid[0]
    best_distance = float("inf")

    for intensity in intensity_grid:
        scenario_weather = build_moderate_weather_scenario(base_weather, intensity, barangay=barangay)
        prediction = predict_with_weather(barangay, registry=registry, weather=scenario_weather)
        day1_prediction = prediction["predictions"][0]
        depth_cm = float(day1_prediction["predicted_depth_cm"])
        risk = str(day1_prediction["risk_level"]).upper()
        distance = _distance_to_target_risk_band(depth_cm, normalized_target)

        if distance < best_distance:
            best_distance = distance
            best_result = prediction
            best_intensity = intensity

        if risk == normalized_target:
            best_result = prediction
            best_intensity = intensity
            best_distance = 0.0
            break

    if best_result is None:
        raise ValueError(f"Unable to generate {normalized_target} risk scenario for {barangay}.")

    day1_prediction = best_result["predictions"][0]
    return {
        "barangay": best_result["barangay"],
        "target_risk_level": normalized_target,
        "selected_intensity": round(best_intensity, 2),
        "achieved_target_risk": str(day1_prediction["risk_level"]).upper() == normalized_target,
        "distance_to_target_cm": round(best_distance, 2),
        "predictions": best_result["predictions"],
        "metrics": best_result["metrics"],
    }


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
    conditional_depth = float(reg_model.predict(X)[0]) if reg_model is not None else 0.0
    conditional_depth = max(0.0, conditional_depth)
    depth = conditional_depth * float(prob)

    return row, float(prob), depth, flood


def _predict_depth_components(model_bundle: dict, X: pd.DataFrame, prob: float) -> dict:
    """Return expected and conditional depth components for a prepared feature row."""

    reg_model = model_bundle.get("reg")
    reg_q50_model = model_bundle.get("reg_q50")
    reg_q90_model = model_bundle.get("reg_q90")
    features = model_bundle["features"]

    conditional_depth = float(reg_model.predict(X[features])[0]) if reg_model is not None else 0.0
    conditional_depth = max(0.0, conditional_depth)
    expected_depth = conditional_depth * float(prob)

    conditional_p50 = conditional_depth
    if reg_q50_model is not None:
        conditional_p50 = max(0.0, float(reg_q50_model.predict(X[features])[0]))

    conditional_p90 = conditional_depth
    if reg_q90_model is not None:
        conditional_p90 = max(0.0, float(reg_q90_model.predict(X[features])[0]))

    return {
        "expected_depth": expected_depth,
        "conditional_depth": conditional_depth,
        "conditional_depth_p50": conditional_p50,
        "conditional_depth_p90": conditional_p90,
    }


def _predict_day1_depth_cm(
    barangay: str,
    scenario_weather: pd.DataFrame,
    registry: Optional[dict] = None,
) -> float:
    """Fast path: score only the first weather row and return the expected depth. Used by grid search."""
    model_bundle = get_model_for_barangay(barangay, registry)
    features = model_bundle["features"]
    row = scenario_weather.iloc[0]
    X = _build_feature_row(row, features, model_bundle["barangay"])
    rf_prob = model_bundle["rf"].predict_proba(X[features])[0][1]
    xgb_prob = model_bundle["xgb"].predict_proba(X[features])[0][1]
    prob = (rf_prob + xgb_prob) / 2
    reg_model = model_bundle.get("reg")
    conditional_depth = max(0.0, float(reg_model.predict(X[features])[0])) if reg_model is not None else 0.0
    return conditional_depth * prob


def predict_with_weather(
    barangay: str,
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
) -> dict:
    """Generate short-term forecasts for a barangay using freshly pulled weather data."""

    model_bundle = get_model_for_barangay(barangay, registry)
    features = model_bundle["features"]
    rf_model = model_bundle["rf"]
    xgb_model = model_bundle["xgb"]
    reg_model = model_bundle.get("reg")
    barangay_metrics = model_bundle["metrics"]

    weather_df = weather if weather is not None else get_weather()
    results = []
    barangay_name = model_bundle["barangay"]

    for i, row in weather_df.iterrows():
        X = _build_feature_row(row, features, barangay_name)

        rf_prob = rf_model.predict_proba(X[features])[0][1]
        xgb_prob = xgb_model.predict_proba(X[features])[0][1]
        prob = (rf_prob + xgb_prob) / 2
        flood = int(prob > CLASSIFICATION_THRESHOLD)
        conditional_depth = float(reg_model.predict(X[features])[0]) if reg_model is not None else 0.0
        conditional_depth = max(0.0, conditional_depth)
        depth = conditional_depth * float(prob)
        rl = risk_level(depth)

        results.append(
            {
                "day": i + 1,
                "flood_probability": round(float(prob), 4),
                "predicted_depth_cm": display_depth_cm(depth),
                "risk_level": rl,
                "alert": flood,
                "summary": format_prediction_summary(prob, depth, rl),
            }
        )

    return {"barangay": model_bundle["barangay"], "predictions": results, "metrics": barangay_metrics}


def predict_with_weather_depth_diagnostics(
    barangay: str,
    registry: Optional[dict[str, dict]] = None,
    weather: Optional[pd.DataFrame] = None,
) -> dict:
    """Return weather prediction output with additional depth diagnostics for testing."""

    model_bundle = get_model_for_barangay(barangay, registry)
    features = model_bundle["features"]
    rf_model = model_bundle["rf"]
    xgb_model = model_bundle["xgb"]
    barangay_metrics = model_bundle["metrics"]

    weather_df = weather if weather is not None else get_weather()
    results = []
    barangay_name = model_bundle["barangay"]

    for i, row in weather_df.iterrows():
        X = _build_feature_row(row, features, barangay_name)

        rf_prob = rf_model.predict_proba(X[features])[0][1]
        xgb_prob = xgb_model.predict_proba(X[features])[0][1]
        prob = (rf_prob + xgb_prob) / 2
        flood = int(prob > CLASSIFICATION_THRESHOLD)

        depth_parts = _predict_depth_components(model_bundle, X, prob)
        expected_depth = depth_parts["expected_depth"]
        rl = risk_level(expected_depth)

        results.append(
            {
                "day": i + 1,
                "flood_probability": round(float(prob), 4),
                "predicted_depth_cm": display_depth_cm(expected_depth),
                "predicted_depth_if_flood_cm": round(float(depth_parts["conditional_depth"]), 2),
                "predicted_depth_if_flood_p50_cm": round(float(depth_parts["conditional_depth_p50"]), 2),
                "predicted_depth_if_flood_p90_cm": round(float(depth_parts["conditional_depth_p90"]), 2),
                "risk_level": rl,
                "alert": flood,
                "summary": format_prediction_summary(prob, expected_depth, rl),
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

    rl = risk_level(depth)
    prediction = {
        "flood_probability": round(prob, 4),
        "predicted_depth_cm": display_depth_cm(depth),
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
