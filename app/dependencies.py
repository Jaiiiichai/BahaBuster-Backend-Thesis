"""Simple dependency container for FastAPI routes."""

from __future__ import annotations

from functools import lru_cache

from .integrations.weather_client import WeatherClient
from .ml.registry import ModelRegistry
from .services.forecast_service import ForecastService
from .services.prediction_service import PredictionService


@lru_cache
def get_weather_client() -> WeatherClient:
    return WeatherClient()


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


@lru_cache
def get_forecast_service() -> ForecastService:
    return ForecastService(get_weather_client())


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService(get_model_registry(), get_forecast_service())
