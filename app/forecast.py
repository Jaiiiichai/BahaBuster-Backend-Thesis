"""Compatibility layer for legacy imports."""

from __future__ import annotations

from .services.forecast_service import ForecastService

_FORECAST_SERVICE = ForecastService()


def build_forecast(latitude: float, longitude: float):
    """Delegate to the new service implementation."""
    return _FORECAST_SERVICE.build_forecast(latitude, longitude)
