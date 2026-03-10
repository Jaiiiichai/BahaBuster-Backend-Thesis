"""Client for retrieving weather forecasts from Open-Meteo."""

from __future__ import annotations

from typing import Dict, Any

import requests


class WeatherClient:
    """Lightweight wrapper around the Open-Meteo forecast API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, session: requests.Session | None = None, forecast_days: int = 3):
        self._session = session or requests.Session()
        self._forecast_days = forecast_days

    def fetch_hourly_forecast(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Return raw hourly forecast data for the provided coordinates."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,rain",
            "forecast_days": self._forecast_days,
        }

        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            raise ValueError(f"Weather API request failed: {exc}") from exc

        payload = response.json()
        hourly = payload.get("hourly")
        if not hourly:
            raise ValueError("Weather API failed to provide hourly data")

        return hourly
