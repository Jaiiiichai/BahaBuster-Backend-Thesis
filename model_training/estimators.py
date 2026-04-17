"""Model-building helpers for classification and regression tasks."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier


def fit_classifiers(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestClassifier, XGBClassifier]:
    """Train the Random Forest and XGBoost classifiers on a balanced dataset."""

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
    )
    rf_model.fit(X_res, y_res)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    xgb_model.fit(X_res, y_res)

    return rf_model, xgb_model


def predict_probabilities(
    rf_model: RandomForestClassifier, xgb_model: XGBClassifier, X: pd.DataFrame
) -> pd.Series:
    """Blend classifier probabilities by averaging the Random Forest and XGBoost scores."""

    rf_prob = rf_model.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    return (rf_prob + xgb_prob) / 2


def fit_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train the Random Forest regressor used for flood depth estimation."""

    reg_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    reg_model.fit(X_train, y_train)
    return reg_model


def fit_quantile_regressor(X_train: pd.DataFrame, y_train: pd.Series, quantile: float) -> GradientBoostingRegressor:
    """Train a quantile regressor for conditional flood-depth distribution estimates."""

    q_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    q_model.fit(X_train, y_train)
    return q_model
