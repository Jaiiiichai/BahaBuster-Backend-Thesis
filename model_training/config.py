"""Shared configuration constants for the model_training package."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "barangay_flood_dataset.csv"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

COLUMN_RENAMES = {
    "FLOOD_OCCURRENCE (0/1)": "FLOOD_OCCURRENCE",
    "flood_depth (cm) (numeric)": "flood_depth",
    "RISK_LEVEL (LOW/MEDIUM/HIGH)": "RISK_LEVEL",
    "rain (mm)": "rain",
    "precipitation (mm)": "precipitation",
    "avg_past_flood_depth (cm)": "avg_past_flood_depth",
}

REQUIRED_COLUMNS = [
    "FLOOD_OCCURRENCE",
    "flood_depth",
    "rain",
    "precipitation",
    "relative_humidity_2m (%)",
]

TARGET_CLASS = "FLOOD_OCCURRENCE"
TARGET_REG = "flood_depth"

CLASSIFICATION_THRESHOLD = 0.05
EMPTY_CONFUSION_MATRIX = {
    "tn": None,
    "fp": None,
    "fn": None,
    "tp": None,
}
DEFAULT_CLASSIFICATION_METRICS = {
    "f1": None,
    "auc": None,
    "precision": None,
    "recall": None,
    "confusion_matrix": EMPTY_CONFUSION_MATRIX.copy(),
}
