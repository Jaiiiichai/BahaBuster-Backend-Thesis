"""Public surface for the model_training package."""

from .exceptions import ModelNotFoundError
from .prediction import get_weather, manual_prediction_response, predict_with_weather
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
    "manual_prediction_response",
    "get_weather",
    "MODEL_REGISTRY",
    "ModelNotFoundError",
]
