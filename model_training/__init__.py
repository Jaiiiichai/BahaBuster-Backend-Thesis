"""Public surface for the model_training package."""

from .exceptions import ModelNotFoundError
from .prediction import (
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
from .registry import (
    MODEL_REGISTRY,
    get_available_barangays,
    get_model_for_barangay,
    get_model_registry,
    refresh_model_registry,
)
from .training import train_model

__all__ = [
    "train_model",
    "refresh_model_registry",
    "get_model_registry",
    "get_available_barangays",
    "get_model_for_barangay",
    "predict_with_weather",
    "predict_with_weather_depth_diagnostics",
    "predict_with_moderate_weather_target",
    "predict_with_target_risk_level",
    "find_shared_intensity",
    "predict_with_shared_intensity",
    "predict_all_shared_intensity",
    "manual_prediction_response",
    "get_weather",
    "MODEL_REGISTRY",
    "ModelNotFoundError",
]
