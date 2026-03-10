"""ML asset loading and caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import joblib

from ..config import (
    OCC_PATH,
    DEPTH_PATH,
    FEATURE_PATH,
    METRIC_PATH,
    THRESHOLD_PATH,
    COORDS_PATH,
)
from ..model_manager import train_models


@dataclass
class ModelRegistry:
    """Stores trained models and metadata, loading from disk when available."""

    occ_models: Dict[str, Any] | None = None
    depth_models: Dict[str, Any] | None = None
    features_map: Dict[str, Any] | None = None
    metrics_map: Dict[str, Any] | None = None
    threshold_map: Dict[str, Any] | None = None
    coords_map: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        (
            self.occ_models,
            self.depth_models,
            self.features_map,
            self.metrics_map,
            self.threshold_map,
            self.coords_map,
        ) = self._load_or_train()

    def _load_or_train(self):
        if OCC_PATH.exists():
            return (
                joblib.load(OCC_PATH),
                joblib.load(DEPTH_PATH),
                joblib.load(FEATURE_PATH),
                joblib.load(METRIC_PATH),
                joblib.load(THRESHOLD_PATH),
                joblib.load(COORDS_PATH),
            )

        registry = train_models()
        joblib.dump(registry[0], OCC_PATH)
        joblib.dump(registry[1], DEPTH_PATH)
        joblib.dump(registry[2], FEATURE_PATH)
        joblib.dump(registry[3], METRIC_PATH)
        joblib.dump(registry[4], THRESHOLD_PATH)
        joblib.dump(registry[5], COORDS_PATH)
        return registry
