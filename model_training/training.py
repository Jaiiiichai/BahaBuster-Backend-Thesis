"""Pipeline logic for fitting barangay-specific models."""
from __future__ import annotations

from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, TARGET_CLASS, TARGET_REG
from .data_loader import load_training_dataframe
from .evaluation import (
    classification_metrics_fallback,
    evaluate_temporal_classification_holdout,
    evaluate_temporal_regression_holdout,
)
from .estimators import fit_classifiers, fit_regressor
from .features import create_features
from .naming import barangay_model_path, normalize_barangay_name


def train_model(force_retrain: bool = False) -> dict[str, dict]:
    """Train models for every barangay and persist them to disk if needed."""

    df = load_training_dataframe().drop_duplicates().reset_index(drop=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        sort_cols = ["barangay", "date"] if "barangay" in df.columns else ["date"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)

    df = create_features(df)
    df[TARGET_CLASS] = pd.to_numeric(df[TARGET_CLASS], errors="coerce")
    df[TARGET_REG] = pd.to_numeric(df[TARGET_REG], errors="coerce")

    drop_cols = [
        "date",
        "barangay",
        "RISK_LEVEL",
        "flood_risk_score",
        "avg_past_flood_depth",
        "flood_frequency_category",
    ]
    features = [c for c in df.columns if c not in drop_cols + [TARGET_CLASS, TARGET_REG]]

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=features + [TARGET_CLASS])
        .reset_index(drop=True)
    )
    if "barangay" not in df.columns:
        raise ValueError("Dataset must include 'barangay' column for barangay-specific training.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict] = {}

    for barangay_name, group in df.groupby("barangay"):
        model_path = barangay_model_path(barangay_name)

        if not force_retrain and model_path.exists():
            try:
                existing = joblib.load(model_path)
                normalized_existing = normalize_barangay_name(existing["barangay"])
                existing["barangay"] = normalized_existing
                registry[normalized_existing] = existing
                print(
                    f"[TRAIN] Reusing existing model for {normalized_existing} (pass force_retrain=True to rebuild)."
                )
                continue
            except Exception:
                print(f"[TRAIN] Existing model for {barangay_name} is invalid. Re-training...")

        barangay_model = train_barangay_model(barangay_name, group.reset_index(drop=True), features)
        if not barangay_model:
            continue
        joblib.dump(barangay_model, model_path)
        registry[barangay_model["barangay"]] = barangay_model
        print(f"[TRAIN] Stored model for {barangay_model['barangay']} -> {model_path.name}")

    if not registry:
        raise ValueError("No barangay models were trained. Check dataset balance and targets.")

    print(f"[TRAIN] Completed training for {len(registry)} barangays.")
    return registry


def train_barangay_model(barangay_name: str, df: pd.DataFrame, features: list[str]) -> Optional[dict]:
    """Fit classifiers and regressors for a single barangay and package the artifacts."""

    normalized_name = normalize_barangay_name(barangay_name)
    print(f"[TRAIN] Barangay={normalized_name} samples={len(df)} floods={int(df[TARGET_CLASS].sum())}")

    X = df[features]
    y = df[TARGET_CLASS]

    if y.nunique() < 2:
        print(f"[TRAIN] Skipped {normalized_name}: not enough target diversity.")
        return None

    class_metrics = evaluate_temporal_classification_holdout(X, y)
    if class_metrics is None:
        class_metrics = classification_metrics_fallback()

    rf_model, xgb_model = fit_classifiers(X, y)

    reg_model = None
    reg_metrics: Dict[str, Optional[float]] = {"mae_cm": None, "r2": None}

    reg_mask = df[TARGET_REG] >= 1
    if reg_mask.sum() > 20:
        reg_df = df.loc[reg_mask].reset_index(drop=True)
        X_reg = reg_df[features]
        y_reg = reg_df[TARGET_REG]

        reg_metrics = evaluate_temporal_regression_holdout(X_reg, y_reg) or reg_metrics
        reg_metrics.setdefault("mae_cm", None)
        reg_metrics.setdefault("r2", None)

        reg_model = fit_regressor(X_reg, y_reg)
    else:
        print(f"[REGRESSION] Skipped for {normalized_name}: insufficient depth samples.")

    metrics = {"classification": class_metrics, "regression": reg_metrics}

    return {
        "barangay": normalized_name,
        "rf": rf_model,
        "xgb": xgb_model,
        "reg": reg_model,
        "features": features,
        "metrics": metrics,
    }
