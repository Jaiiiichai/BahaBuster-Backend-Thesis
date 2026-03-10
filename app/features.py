import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["rain_3day"] = df["rain (mm)"].rolling(3, min_periods=1).sum()
    df["rain_7day"] = df["rain (mm)"].rolling(7, min_periods=1).sum()
    df["rain_14day"] = df["rain (mm)"].rolling(14, min_periods=1).sum()
    df["rain_mean_3"] = df["rain (mm)"].rolling(3, min_periods=1).mean()
    df["rain_lag1"] = df["rain (mm)"].shift(1).fillna(0)

    if "atmospheric_pressure (hPa)" in df.columns:
        df["pressure_trend"] = df["atmospheric_pressure (hPa)"].diff().fillna(0)
    else:
        df["pressure_trend"] = 0

    df["soil_saturation"] = (
        df["relative_humidity_2m (%)"] * df["rain_7day"] / 100
    )

    df["rain_intensity_ratio"] = df["rain (mm)"] / (df["rain_3day"] + 1)

    if "elevation (m)" in df.columns:
        df["rain_elev_interaction"] = df["rain_3day"] * df["elevation (m)"]
    else:
        df["rain_elev_interaction"] = 0

    if "FLOOD_OCCURRENCE (0/1)" in df.columns:
        df["flood_lag1"] = df["FLOOD_OCCURRENCE (0/1)"].shift(1).fillna(0)
        df["flood_lag3_sum"] = df["FLOOD_OCCURRENCE (0/1)"].rolling(3).sum().fillna(0)

    if "flood_depth (cm)" in df.columns:
        df["depth_lag1"] = df["flood_depth (cm)"].shift(1).fillna(0)
        df["depth_rolling_mean"] = df["flood_depth (cm)"].rolling(3).mean().fillna(0)
    else:
        df["depth_lag1"] = 0
        df["depth_rolling_mean"] = 0

    return df.fillna(0)
