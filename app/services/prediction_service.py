"""Business logic for running flood predictions."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..ml.registry import ModelRegistry
from ..utils import clean, get_risk_level

from .forecast_service import ForecastService


class PredictionService:
    """Coordinates ML models and weather data to serve predictions."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        forecast_service: ForecastService,
    ) -> None:
        self._registry = model_registry
        self._forecast_service = forecast_service

    def predict_flood(self, barangay: str) -> Dict[str, Any]:
        barangay_key = barangay.upper()
        if barangay_key not in self._registry.occ_models:
            raise ValueError("Barangay not trained")
        if barangay_key not in self._registry.coords_map:
            raise ValueError("Coordinates not available")

        latitude = self._registry.coords_map[barangay_key]["latitude"]
        longitude = self._registry.coords_map[barangay_key]["longitude"]

        try:
            X_pred = self._forecast_service.build_forecast(latitude, longitude)
        except Exception as exc:  # pragma: no cover - external dependency
            raise ValueError(f"Forecast failed: {exc}") from exc

        features = self._registry.features_map[barangay_key]
        X_pred = X_pred.reindex(columns=features, fill_value=0)

        clf = self._registry.occ_models[barangay_key]
        probabilities = clf.predict_proba(X_pred)[:, 1]
        threshold = self._registry.threshold_map.get(barangay_key, 0.5)
        depth_predictions = self._predict_depths(barangay_key, X_pred)

        results = []
        for idx, probability in enumerate(probabilities):
            prob_percent = probability * 100
            results.append(
                {
                    "day": idx + 1,
                    "flood_probability": clean(prob_percent),
                    "predicted_depth_cm": clean(depth_predictions[idx]),
                    "alert": int(probability > threshold),
                    "risk_level": get_risk_level(prob_percent),
                }
            )

        return {
            "barangay": barangay_key,
            "latitude": latitude,
            "longitude": longitude,
            "predictions": results,
            "metrics": self._registry.metrics_map[barangay_key],
        }

    def _predict_depths(self, barangay: str, X_pred: pd.DataFrame) -> List[float]:
        if barangay not in self._registry.depth_models:
            return [0.0] * len(X_pred)

        reg = self._registry.depth_models[barangay]
        depth_values = reg.predict(X_pred)
        return depth_values.tolist()
