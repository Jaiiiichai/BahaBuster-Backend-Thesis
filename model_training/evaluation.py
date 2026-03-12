"""Evaluation helpers for both classification and regression models."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

from .config import CLASSIFICATION_THRESHOLD, DEFAULT_CLASSIFICATION_METRICS, EMPTY_CONFUSION_MATRIX
from .estimators import fit_classifiers, fit_regressor, predict_probabilities


def classification_metrics_fallback() -> dict:
    """Return a safe placeholder when meaningful classification metrics cannot be computed."""

    metrics = DEFAULT_CLASSIFICATION_METRICS.copy()
    metrics["confusion_matrix"] = EMPTY_CONFUSION_MATRIX.copy()
    return metrics


def compute_classification_metrics(y_true: pd.Series, prob: np.ndarray) -> dict:
    """Evaluate core classification metrics using the calibrated probability outputs."""

    pred = (prob > CLASSIFICATION_THRESHOLD).astype(int)
    scores = {
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }

    try:
        scores["auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        scores["auc"] = None

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        scores["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    except ValueError:
        scores["confusion_matrix"] = EMPTY_CONFUSION_MATRIX.copy()

    scalar_scores = {
        k: (None if v is None else float(v))
        for k, v in scores.items()
        if k != "confusion_matrix"
    }
    scalar_scores["confusion_matrix"] = scores["confusion_matrix"]
    return scalar_scores


def aggregate_classification_scores(scores: List[dict]) -> dict:
    """Average metrics across folds while keeping confusion-matrix components additive."""

    if not scores:
        return DEFAULT_CLASSIFICATION_METRICS.copy()

    aggregated: Dict[str, Optional[float]] = {}
    for key in ["f1", "precision", "recall", "auc"]:
        values = [s.get(key) for s in scores if s.get(key) is not None]
        aggregated[key] = None if not values else float(np.mean(values))

    cm_totals = {key: 0 for key in ["tn", "fp", "fn", "tp"]}
    cm_found = False
    for entry in scores:
        cm = entry.get("confusion_matrix")
        if not cm:
            continue
        cm_found = True
        for key in cm_totals:
            value = cm.get(key)
            if value is not None:
                cm_totals[key] += int(value)

    aggregated["confusion_matrix"] = cm_totals if cm_found else EMPTY_CONFUSION_MATRIX.copy()
    return aggregated


def evaluate_group_classification(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: Optional[str],
) -> Optional[dict]:
    """Run grouped cross-validation to understand classification stability per barangay."""

    if not group_col or group_col not in df.columns or df[group_col].nunique() < 2:
        return None

    splitter = GroupKFold(n_splits=min(5, df[group_col].nunique()))
    fold_scores: List[dict] = []

    for train_idx, test_idx in splitter.split(df, groups=df[group_col]):
        y_train = df.loc[train_idx, target_col]
        y_test = df.loc[test_idx, target_col]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        X_train = df.loc[train_idx, features]
        X_test = df.loc[test_idx, features]

        rf_model, xgb_model = fit_classifiers(X_train, y_train)
        prob = predict_probabilities(rf_model, xgb_model, X_test)
        fold_scores.append(compute_classification_metrics(y_test, prob))

    if not fold_scores:
        return None

    return aggregate_classification_scores(fold_scores)


def evaluate_temporal_classification_holdout(X: pd.DataFrame, y: pd.Series) -> Optional[dict]:
    """Evaluate the classifier by keeping the temporal ordering intact during the split."""

    if y.nunique() < 2 or len(y) < 5:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    rf_model, xgb_model = fit_classifiers(X_train, y_train)
    prob = predict_probabilities(rf_model, xgb_model, X_test)
    return compute_classification_metrics(y_test, prob)


def evaluate_group_regression(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: Optional[str],
) -> Optional[dict]:
    """Run grouped cross-validation for the regression head, mirroring deployment splits."""

    if not group_col or group_col not in df.columns or df[group_col].nunique() < 2:
        return None

    splitter = GroupKFold(n_splits=min(5, df[group_col].nunique()))
    maes: List[float] = []
    r2_scores: List[float] = []

    for train_idx, test_idx in splitter.split(df, groups=df[group_col]):
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, features]
        y_test = df.loc[test_idx, target_col]

        if len(y_train) < 2 or len(y_test) == 0:
            continue

        reg_model = fit_regressor(X_train, y_train)
        depth_pred = reg_model.predict(X_test)
        maes.append(mean_absolute_error(y_test, depth_pred))

        try:
            r2_scores.append(r2_score(y_test, depth_pred))
        except ValueError:
            pass

    if not maes and not r2_scores:
        return None

    result: Dict[str, Optional[float]] = {}
    if maes:
        result["mae_cm"] = float(np.mean(maes))
    if r2_scores:
        result["r2"] = float(np.mean(r2_scores))
    return result if result else None


def evaluate_temporal_regression_holdout(X: pd.DataFrame, y: pd.Series) -> Optional[dict]:
    """Hold out the latest samples to gauge regression error on unseen timesteps."""

    if len(y) < 5:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if len(y_train) < 2 or len(y_test) == 0:
        return None

    reg_model = fit_regressor(X_train, y_train)
    depth_pred = reg_model.predict(X_test)

    result = {"mae_cm": float(mean_absolute_error(y_test, depth_pred))}
    try:
        result["r2"] = float(r2_score(y_test, depth_pred))
    except ValueError:
        result["r2"] = None

    return result
