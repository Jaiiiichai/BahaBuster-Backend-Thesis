import pandas as pd
import numpy as np
import requests
import joblib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupKFold

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

app = FastAPI(title="Flood Prediction System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "barangay_flood_dataset.csv"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

COLUMN_RENAMES = {
    "FLOOD_OCCURRENCE (0/1)": "FLOOD_OCCURRENCE",
    "flood_depth (cm) (numeric)": "flood_depth",
    "RISK_LEVEL (LOW/MEDIUM/HIGH)": "RISK_LEVEL",
    "rain (mm)": "rain",
    "precipitation (mm)": "precipitation",
    "avg_past_flood_depth (cm)": "avg_past_flood_depth"
}

REQUIRED_COLUMNS = [
    "FLOOD_OCCURRENCE",
    "flood_depth",
    "rain",
    "precipitation",
    "relative_humidity_2m (%)"
]

TARGET_CLASS = "FLOOD_OCCURRENCE"
TARGET_REG = "flood_depth"

CLASSIFICATION_THRESHOLD = 0.05
EMPTY_CONFUSION_MATRIX = {
    "tn": None,
    "fp": None,
    "fn": None,
    "tp": None
}
DEFAULT_CLASSIFICATION_METRICS = {
    "f1": None,
    "auc": None,
    "precision": None,
    "recall": None,
    "confusion_matrix": EMPTY_CONFUSION_MATRIX.copy()
}


def _normalize_barangay_name(name: Optional[str]) -> str:
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


def _slugify_barangay_name(name: str) -> str:
    normalized = _normalize_barangay_name(name)
    return normalized.lower().replace(" ", "_")


def _barangay_model_path(name: str) -> Path:
    return MODEL_DIR / f"{_slugify_barangay_name(name)}_model.pkl"


def _classification_metrics_fallback() -> dict:

    metrics = DEFAULT_CLASSIFICATION_METRICS.copy()
    metrics["confusion_matrix"] = EMPTY_CONFUSION_MATRIX.copy()
    return metrics


class ManualPredictionRequest(BaseModel):
    barangay: Optional[str] = None
    features: Dict[str, float]
    actual_flood: Optional[bool] = None
    actual_depth_cm: Optional[float] = None

# -------------------------
# LOAD DATA
# -------------------------
def _read_training_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    header_row = 0 if "FLOOD_OCCURRENCE" in first_line.upper() else 1
    return pd.read_csv(csv_path, header=header_row)

def _standardize_training_columns(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns=COLUMN_RENAMES)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in dataset {source}")
    return df

def load_training_dataframe() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = _standardize_training_columns(_read_training_csv(DATA_PATH), DATA_PATH)
        if "barangay" not in df.columns:
            raise ValueError(f"Dataset {DATA_PATH.name} must include a 'barangay' column for barangay-specific training.")
        df["barangay"] = df["barangay"].apply(_normalize_barangay_name)
        return df
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Training data not found. Neither {DATA_PATH.name} nor {DATA_DIR} exists.")
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Training data not found. Place a CSV in {DATA_DIR} or {DATA_PATH.name}.")
    frames = []
    for csv_path in csv_files:
        df = _standardize_training_columns(_read_training_csv(csv_path), csv_path)
        if "barangay" not in df.columns:
            df["barangay"] = _normalize_barangay_name(csv_path.stem)
        else:
            df["barangay"] = df["barangay"].apply(_normalize_barangay_name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "barangay" in df.columns:
        df["rain_lag1"] = df.groupby("barangay")["rain"].shift(1).fillna(0)
        df["rain_lag2"] = df.groupby("barangay")["rain"].shift(2).fillna(0)
    else:
        df["rain_lag1"] = df["rain"].shift(1).fillna(0)
        df["rain_lag2"] = df["rain"].shift(2).fillna(0)
    return df


def _fit_classifiers(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestClassifier, XGBClassifier]:
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X_res, y_res)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model.fit(X_res, y_res)

    return rf_model, xgb_model


def _predict_probabilities(rf_model: RandomForestClassifier, xgb_model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    rf_prob = rf_model.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    return (rf_prob + xgb_prob) / 2


def _classification_metrics(y_true: pd.Series, prob: np.ndarray) -> dict:
    pred = (prob > CLASSIFICATION_THRESHOLD).astype(int)

    scores = {
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0)
    }

    try:
        scores["auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        scores["auc"] = None

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        scores["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    except ValueError:
        scores["confusion_matrix"] = EMPTY_CONFUSION_MATRIX.copy()

    scalar_scores = {k: (None if v is None else float(v)) for k, v in scores.items() if k != "confusion_matrix"}
    scalar_scores["confusion_matrix"] = scores["confusion_matrix"]
    return scalar_scores


def _aggregate_classification_scores(scores: list[dict]) -> dict:

    if not scores:
        return DEFAULT_CLASSIFICATION_METRICS.copy()

    aggregated = {}
    simple_keys = ["f1", "precision", "recall", "auc"]

    for key in simple_keys:
        values = [s.get(key) for s in scores if s.get(key) is not None]
        aggregated[key] = None if not values else float(np.mean(values))

    cm_keys = ["tn", "fp", "fn", "tp"]
    cm_totals = {k: 0 for k in cm_keys}
    cm_found = False

    for entry in scores:
        cm = entry.get("confusion_matrix")
        if not cm:
            continue
        cm_found = True
        for key in cm_keys:
            value = cm.get(key)
            if value is not None:
                cm_totals[key] += int(value)

    aggregated["confusion_matrix"] = cm_totals if cm_found else EMPTY_CONFUSION_MATRIX.copy()

    return aggregated


def evaluate_group_classification(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    group_col: str | None
) -> dict | None:

    if not group_col or group_col not in df.columns or df[group_col].nunique() < 2:
        return None

    splitter = GroupKFold(n_splits=min(5, df[group_col].nunique()))
    fold_scores = []

    for train_idx, test_idx in splitter.split(df, groups=df[group_col]):
        y_train = df.loc[train_idx, target_col]
        y_test = df.loc[test_idx, target_col]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        X_train = df.loc[train_idx, features]
        X_test = df.loc[test_idx, features]

        rf_model, xgb_model = _fit_classifiers(X_train, y_train)
        prob = _predict_probabilities(rf_model, xgb_model, X_test)

        fold_scores.append(_classification_metrics(y_test, prob))

    if not fold_scores:
        return None

    return _aggregate_classification_scores(fold_scores)


def evaluate_temporal_classification_holdout(
    X: pd.DataFrame,
    y: pd.Series
) -> dict | None:

    if y.nunique() < 2 or len(y) < 5:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    rf_model, xgb_model = _fit_classifiers(X_train, y_train)
    prob = _predict_probabilities(rf_model, xgb_model, X_test)

    return _classification_metrics(y_test, prob)


def _fit_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    reg_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    reg_model.fit(X_train, y_train)
    return reg_model


def evaluate_group_regression(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    group_col: str | None
) -> dict | None:

    if not group_col or group_col not in df.columns or df[group_col].nunique() < 2:
        return None

    splitter = GroupKFold(n_splits=min(5, df[group_col].nunique()))
    maes: list[float] = []
    r2_scores: list[float] = []

    for train_idx, test_idx in splitter.split(df, groups=df[group_col]):
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, features]
        y_test = df.loc[test_idx, target_col]

        if len(y_train) < 2 or len(y_test) == 0:
            continue

        reg_model = _fit_regressor(X_train, y_train)
        depth_pred = reg_model.predict(X_test)
        maes.append(mean_absolute_error(y_test, depth_pred))

        try:
            r2_scores.append(r2_score(y_test, depth_pred))
        except ValueError:
            pass

    if not maes and not r2_scores:
        return None

    result = {}
    if maes:
        result["mae_cm"] = float(np.mean(maes))
    if r2_scores:
        result["r2"] = float(np.mean(r2_scores))

    return result if result else None


def evaluate_temporal_regression_holdout(
    X: pd.DataFrame,
    y: pd.Series
) -> dict | None:

    if len(y) < 5:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    if len(y_train) < 2 or len(y_test) == 0:
        return None

    reg_model = _fit_regressor(X_train, y_train)
    depth_pred = reg_model.predict(X_test)

    result = {
        "mae_cm": float(mean_absolute_error(y_test, depth_pred))
    }

    try:
        result["r2"] = float(r2_score(y_test, depth_pred))
    except ValueError:
        result["r2"] = None

    return result

# -------------------------
# TRAIN MODEL
# -------------------------
def train_model(force_retrain: bool = False):
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

    DROP = [
        "date", "barangay", "RISK_LEVEL", "flood_risk_score", "avg_past_flood_depth",
        "flood_frequency_category"
    ]
    FEATURES = [c for c in df.columns if c not in DROP + [TARGET_CLASS, TARGET_REG]]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET_CLASS]).reset_index(drop=True)
    if "barangay" not in df.columns:
        raise ValueError("Dataset must include 'barangay' column for barangay-specific training.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict] = {}

    for barangay_name, group in df.groupby("barangay"):
        model_path = _barangay_model_path(barangay_name)

        if not force_retrain and model_path.exists():
            try:
                existing = joblib.load(model_path)
                normalized_existing = _normalize_barangay_name(existing["barangay"])
                existing["barangay"] = normalized_existing
                registry[normalized_existing] = existing
                print(f"[TRAIN] Reusing existing model for {normalized_existing} (pass force_retrain=True to rebuild).")
                continue
            except Exception:
                print(f"[TRAIN] Existing model for {barangay_name} is invalid. Re-training...")

        barangay_model = _train_barangay_model(barangay_name, group.reset_index(drop=True), FEATURES)
        if not barangay_model:
            continue
        joblib.dump(barangay_model, model_path)
        registry[barangay_model["barangay"]] = barangay_model
        print(f"[TRAIN] Stored model for {barangay_model['barangay']} -> {model_path.name}")

    if not registry:
        raise ValueError("No barangay models were trained. Check dataset balance and targets.")

    print(f"[TRAIN] Completed training for {len(registry)} barangays.")
    return registry


def _train_barangay_model(barangay_name: str, df: pd.DataFrame, features: list[str]) -> Optional[dict]:
    normalized_name = _normalize_barangay_name(barangay_name)
    print(f"[TRAIN] Barangay={normalized_name} samples={len(df)} floods={int(df[TARGET_CLASS].sum())}")

    X = df[features]
    y = df[TARGET_CLASS]

    if y.nunique() < 2:
        print(f"[TRAIN] Skipped {normalized_name}: not enough target diversity.")
        return None

    class_metrics = evaluate_temporal_classification_holdout(X, y)
    if class_metrics is None:
        class_metrics = _classification_metrics_fallback()

    rf_model, xgb_model = _fit_classifiers(X, y)

    reg_model = None
    reg_metrics = {"mae_cm": None, "r2": None}

    reg_mask = df[TARGET_REG] >= 1
    if reg_mask.sum() > 20:
        reg_df = df.loc[reg_mask].reset_index(drop=True)
        X_reg = reg_df[features]
        y_reg = reg_df[TARGET_REG]

        reg_metrics = evaluate_temporal_regression_holdout(X_reg, y_reg) or reg_metrics
        reg_metrics.setdefault("mae_cm", None)
        reg_metrics.setdefault("r2", None)

        reg_model = _fit_regressor(X_reg, y_reg)
    else:
        print(f"[REGRESSION] Skipped for {normalized_name}: insufficient depth samples.")

    metrics = {
        "classification": class_metrics,
        "regression": reg_metrics
    }

    return {
        "barangay": normalized_name,
        "rf": rf_model,
        "xgb": xgb_model,
        "reg": reg_model,
        "features": features,
        "metrics": metrics
    }


def _load_model_registry() -> dict[str, dict]:
    try:
        return train_model()
    except Exception as training_exc:
        print(f"[LOAD] Training pipeline unavailable ({training_exc}). Falling back to cached models.")

    registry: dict[str, dict] = {}
    if MODEL_DIR.exists():
        for model_file in MODEL_DIR.glob("*_model.pkl"):
            try:
                bundle = joblib.load(model_file)
            except Exception as exc:
                print(f"[LOAD] Skipping {model_file.name}: {exc}")
                continue
            barangay_name = bundle.get("barangay")
            if not barangay_name:
                continue
            try:
                normalized_name = _normalize_barangay_name(barangay_name)
            except ValueError:
                continue
            bundle["barangay"] = normalized_name
            registry[normalized_name] = bundle

    if registry:
        return registry

    raise RuntimeError("No trained models available. Provide training data and rerun train_model().")


MODEL_REGISTRY = _load_model_registry()


def _get_model_for_barangay(barangay: str) -> dict:
    try:
        normalized = _normalize_barangay_name(barangay)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model_bundle = MODEL_REGISTRY.get(normalized)
    if not model_bundle:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{barangay}'. Available: {available}."
        )
    return model_bundle

# RISK LEVEL
# -------------------------
def risk_level(p):
    if p < 0.1: return "LOW"
    elif p < 0.3: return "MODERATE"
    elif p < 0.6: return "HIGH"
    else: return "SEVERE"


def format_prediction_summary(prob: float, depth: float, level: str) -> str:
    return (
        f"Flood probability {prob*100:.1f}%, risk level {level}, "
        f"expected depth {depth:.1f} cm."
    )

# -------------------------
# GET WEATHER
# -------------------------
def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.3157,
        "longitude": 123.8854,
        "daily": "temperature_2m_max,relative_humidity_2m_max,precipitation_sum",
        "timezone": "auto"
    }
    res = requests.get(url, params=params)
    data = res.json()["daily"]

    df = pd.DataFrame({
        "temperature_2m (°C)": data["temperature_2m_max"],
        "relative_humidity_2m (%)": data["relative_humidity_2m_max"],
        "rain": data["precipitation_sum"]
    })
    df["precipitation"] = df["rain"]
    df = create_features(df)
    return df.head(3)


def _run_manual_prediction(feature_map: Dict[str, float], model_bundle: dict) -> tuple[Dict[str, float], float, float, int]:

    features = model_bundle["features"]
    rf_model = model_bundle["rf"]
    xgb_model = model_bundle["xgb"]
    reg_model = model_bundle.get("reg")

    row = {feature: float(feature_map.get(feature, 0.0)) for feature in features}

    X = pd.DataFrame([row])

    rf_prob = rf_model.predict_proba(X)[0][1]
    xgb_prob = xgb_model.predict_proba(X)[0][1]
    prob = (rf_prob + xgb_prob) / 2

    flood = int(prob > CLASSIFICATION_THRESHOLD)

    if reg_model is not None and flood == 1:
        depth = float(reg_model.predict(X)[0])
    else:
        depth = 0.0

    return row, float(prob), depth, flood

# -------------------------
# PREDICTION
# -------------------------
def _predict_response(barangay: str):
    model_bundle = _get_model_for_barangay(barangay)
    features = model_bundle["features"]
    rf_model = model_bundle["rf"]
    xgb_model = model_bundle["xgb"]
    reg_model = model_bundle.get("reg")
    barangay_metrics = model_bundle["metrics"]

    weather = get_weather()
    results = []

    for i, row in weather.iterrows():
        # Only keep features that exist in the row
        valid_features = [f for f in features if f in row.index]
        X = row[valid_features].to_frame().T

        # Fill any missing columns with 0 (just in case)
        for f in features:
            if f not in X.columns:
                X[f] = 0

        rf_prob = rf_model.predict_proba(X[features])[0][1]
        xgb_prob = xgb_model.predict_proba(X[features])[0][1]
        prob = (rf_prob + xgb_prob) / 2
        flood = int(prob > CLASSIFICATION_THRESHOLD)
        depth = float(reg_model.predict(X[features])[0]) if reg_model is not None and flood == 1 else 0
        rl = risk_level(prob)

        results.append({
            "day": i+1,
            "flood_probability": round(float(prob), 4),
            "predicted_depth_cm": round(float(depth), 2),
            "risk_level": rl,
            "alert": flood,
            "summary": format_prediction_summary(prob, depth, rl)
        })

    return {"barangay": model_bundle["barangay"], "predictions": results, "metrics": barangay_metrics}

@app.get("/predict/{barangay}")
def predict(barangay: str):
    return _predict_response(barangay)

@app.get("/predict_flood")
def predict_flood(barangay: str = Query(..., description="Barangay name")):
    return _predict_response(barangay)


@app.post("/predict_manual")
def predict_manual(payload: ManualPredictionRequest):

    if not payload.features:
        raise HTTPException(status_code=400, detail="'features' must contain at least one numeric value.")

    if not payload.barangay:
        raise HTTPException(status_code=400, detail="'barangay' is required for manual predictions.")

    model_bundle = _get_model_for_barangay(payload.barangay)

    feature_row, prob, depth, flood = _run_manual_prediction(payload.features, model_bundle)

    actuals = None

    if payload.actual_flood is not None or payload.actual_depth_cm is not None:

        actuals = {
            "flood_occurred": payload.actual_flood,
            "depth_cm": payload.actual_depth_cm,
            "is_alert_correct": None if payload.actual_flood is None else (bool(flood) == bool(payload.actual_flood))
        }

        if payload.actual_depth_cm is not None:
            actuals["depth_error_cm"] = None if depth is None else round(payload.actual_depth_cm - depth, 2)

    rl = risk_level(prob)
    prediction = {
        "flood_probability": round(prob, 4),
        "predicted_depth_cm": round(depth, 2),
        "risk_level": rl,
        "alert": flood,
        "summary": format_prediction_summary(prob, depth, rl)
    }

    return {
        "barangay": model_bundle["barangay"],
        "prediction": prediction,
        "input_features": feature_row,
        "actuals": actuals,
        "metrics": model_bundle["metrics"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)