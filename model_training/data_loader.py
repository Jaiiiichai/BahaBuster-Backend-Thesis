"""Utilities for reading and standardizing barangay training datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import COLUMN_RENAMES, DATA_DIR, DATA_PATH, REQUIRED_COLUMNS
from .naming import normalize_barangay_name


def read_training_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file and auto-detect whether it contains header metadata rows."""

    with csv_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    header_row = 0 if "FLOOD_OCCURRENCE" in first_line.upper() else 1
    return pd.read_csv(csv_path, header=header_row)


def standardize_training_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Rename and validate dataset columns so downstream steps see consistent inputs."""

    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns=COLUMN_RENAMES)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in dataset {source}")
    return df


def load_training_dataframe() -> pd.DataFrame:
    """Return a normalized DataFrame built from the single CSV or all files in /data."""

    if DATA_PATH.exists():
        df = standardize_training_columns(read_training_csv(DATA_PATH), DATA_PATH)
        if "barangay" not in df.columns:
            raise ValueError(
                f"Dataset {DATA_PATH.name} must include a 'barangay' column for barangay-specific training."
            )
        df["barangay"] = df["barangay"].apply(normalize_barangay_name)
        return df

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Training data not found. Neither {DATA_PATH.name} nor {DATA_DIR} exists.")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Training data not found. Place a CSV in {DATA_DIR} or {DATA_PATH.name}.")

    frames = []
    for csv_path in csv_files:
        df = standardize_training_columns(read_training_csv(csv_path), csv_path)
        if "barangay" not in df.columns:
            df["barangay"] = normalize_barangay_name(csv_path.stem)
        else:
            df["barangay"] = df["barangay"].apply(normalize_barangay_name)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
