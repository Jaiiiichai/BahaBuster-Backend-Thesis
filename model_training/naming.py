"""Helpers for normalizing and referencing barangay-specific assets."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import MODEL_DIR


def normalize_barangay_name(name: Optional[str]) -> str:
    """Normalize raw barangay labels so lookups remain consistent."""

    if name is None or pd.isna(name):
        raise ValueError("Barangay name is required.")
    if not isinstance(name, str):
        name = str(name)

    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Barangay name is required.")

    normalized = " ".join(cleaned.split()).upper()
    normalized = re.sub(r"\s*[-_]*\s*DATA$", "", normalized).strip()
    if not normalized:
        raise ValueError("Barangay name is required.")
    return normalized


def slugify_barangay_name(name: str) -> str:
    """Generate a filesystem-friendly slug for the provided barangay."""

    normalized = normalize_barangay_name(name)
    return normalized.lower().replace(" ", "_")


def barangay_model_path(name: str) -> Path:
    """Return the absolute path where the barangay model pickle should live."""

    return MODEL_DIR / f"{slugify_barangay_name(name)}_model.pkl"
