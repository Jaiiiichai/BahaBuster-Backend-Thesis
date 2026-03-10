"""Transforms raw weather data into model-ready features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import engineer_features
from ..integrations.weather_client import WeatherClient


class ForecastService:
    """Responsible for converting weather API data into ML features."""

    def __init__(self, weather_client: WeatherClient | None = None, forecast_days: int = 3):
        self._weather_client = weather_client or WeatherClient(forecast_days=forecast_days)
        self._forecast_days = forecast_days

    def build_forecast(self, latitude: float, longitude: float) -> pd.DataFrame:
        hourly = self._weather_client.fetch_hourly_forecast(latitude, longitude)
        df = pd.DataFrame(
            {
                "temperature_2m (°C)": hourly["temperature_2m"],
                "relative_humidity_2m (%)": hourly["relative_humidity_2m"],
                "rain (mm)": hourly["rain"],
            }
        )

        df["day"] = self._build_day_index(len(df))
        daily = (
            df.groupby("day")
            .agg(
                {
                    "temperature_2m (°C)": "mean",
                    "relative_humidity_2m (%)": "mean",
                    "rain (mm)": "sum",
                }
            )
            .reset_index(drop=True)
        )

        return engineer_features(daily)

    def _build_day_index(self, length: int) -> np.ndarray:
        """Map hourly samples to forecast days without dropping leftovers."""
        if length == 0:
            return np.array([], dtype=int)

        indices = np.arange(length)
        day_chunks = np.array_split(indices, self._forecast_days)
        day_labels = np.zeros(length, dtype=int)
        for day, chunk in enumerate(day_chunks, start=1):
            day_labels[chunk] = day
        return day_labels
