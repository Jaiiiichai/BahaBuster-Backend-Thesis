import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    DATA_DIR,
    OCC_PATH,
    DEPTH_PATH,
    FEATURE_PATH,
    METRIC_PATH,
    THRESHOLD_PATH,
    COORDS_PATH,
)
from .features import engineer_features
from .utils import clean


def train_models():
    occ_models = {}
    depth_models = {}
    features_map = {}
    metrics_map = {}
    threshold_map = {}
    coords_map = {}

    for file in DATA_DIR.iterdir():
        if file.suffix.lower() != ".csv":
            continue

        barangay = file.name.replace("-DATA.csv", "").upper()
        print("Training:", barangay)

        df = pd.read_csv(file, header=1)
        df = df.ffill()

        if "latitude" not in df.columns or "longitude" not in df.columns:
            print(f"⚠ No coordinates for {barangay}")
            continue

        coords_map[barangay] = {
            "latitude": float(df["latitude"].iloc[0]),
            "longitude": float(df["longitude"].iloc[0]),
        }

        df = engineer_features(df)

        if len(df) < 500:
            continue

        features = [
            "temperature_2m (°C)",
            "relative_humidity_2m (%)",
            "rain (mm)",
            "rain_3day",
            "rain_7day",
            "rain_14day",
            "rain_mean_3",
            "rain_lag1",
            "soil_saturation",
            "rain_intensity_ratio",
            "pressure_trend",
            "rain_elev_interaction",
            "flood_lag1",
            "flood_lag3_sum",
            "depth_lag1",
            "depth_rolling_mean",
        ]

        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        features_map[barangay] = features

        X = df[features]
        y_occ = df["FLOOD_OCCURRENCE (0/1)"].astype(int)
        y_depth = df["flood_depth (cm)"] if "flood_depth (cm)" in df.columns else None

        pos_weight = (len(y_occ) - sum(y_occ)) / max(sum(y_occ), 1)
        tscv = TimeSeriesSplit(n_splits=5)

        f1_scores, auc_scores, best_thresholds = [], [], []

        for train_idx, test_idx in tscv.split(X):
            y_test = y_occ.iloc[test_idx]
            if len(np.unique(y_test)) < 2:
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y_occ.iloc[train_idx]

            clf = XGBClassifier(
                n_estimators=1500,
                learning_rate=0.01,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=42,
            )

            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)[:, 1]

            thresholds = np.arange(0.1, 0.9, 0.01)
            scores = [
                f1_score(y_test, (prob > t).astype(int), zero_division=0)
                for t in thresholds
            ]

            best_threshold = thresholds[int(np.argmax(scores))]
            best_thresholds.append(best_threshold)

            final_pred = (prob > best_threshold).astype(int)
            f1_scores.append(f1_score(y_test, final_pred, zero_division=0))
            auc_scores.append(roc_auc_score(y_test, prob))

        if len(f1_scores) == 0:
            continue

        final_clf = clf.fit(X, y_occ)
        occ_models[barangay] = final_clf
        threshold_map[barangay] = float(np.mean(best_thresholds))

        y_pred_full = final_clf.predict(X)
        r2_class = r2_score(y_occ, y_pred_full)
        cm = confusion_matrix(y_occ, y_pred_full).tolist()

        full_mae = None
        if y_depth is not None:
            flood_mask = y_depth > 0
            if flood_mask.sum() > 0:
                reg = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.01,
                    max_depth=8,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                )
                reg.fit(X[flood_mask], y_depth[flood_mask])
                depth_models[barangay] = reg
                full_mae = mean_absolute_error(
                    y_depth[flood_mask], reg.predict(X[flood_mask])
                )

        metrics_map[barangay] = {
            "classification": {
                "f1": clean(np.mean(f1_scores)),
                "auc": clean(np.mean(auc_scores)),
                "r2": clean(r2_class),
                "confusion_matrix": cm,
            },
            "regression": {
                "mae_cm": clean(full_mae),
            },
        }

    return (
        occ_models,
        depth_models,
        features_map,
        metrics_map,
        threshold_map,
        coords_map,
    )


def load_or_train_models():
    if OCC_PATH.exists():
        occ_models = joblib.load(OCC_PATH)
        depth_models = joblib.load(DEPTH_PATH)
        features_map = joblib.load(FEATURE_PATH)
        metrics_map = joblib.load(METRIC_PATH)
        threshold_map = joblib.load(THRESHOLD_PATH)
        coords_map = joblib.load(COORDS_PATH)
    else:
        (
            occ_models,
            depth_models,
            features_map,
            metrics_map,
            threshold_map,
            coords_map,
        ) = train_models()
        joblib.dump(occ_models, OCC_PATH)
        joblib.dump(depth_models, DEPTH_PATH)
        joblib.dump(features_map, FEATURE_PATH)
        joblib.dump(metrics_map, METRIC_PATH)
        joblib.dump(threshold_map, THRESHOLD_PATH)
        joblib.dump(coords_map, COORDS_PATH)

    return (
        occ_models,
        depth_models,
        features_map,
        metrics_map,
        threshold_map,
        coords_map,
    )
