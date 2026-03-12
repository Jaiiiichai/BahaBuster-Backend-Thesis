"""Feature engineering helpers."""
from __future__ import annotations

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rainfall lag features, grouping by barangay when that column exists."""

    df = df.copy()
    if "barangay" in df.columns:
        df["rain_lag1"] = df.groupby("barangay")["rain"].shift(1).fillna(0)
        df["rain_lag2"] = df.groupby("barangay")["rain"].shift(2).fillna(0)
    else:
        df["rain_lag1"] = df["rain"].shift(1).fillna(0)
        df["rain_lag2"] = df["rain"].shift(2).fillna(0)
    return df
