from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OCC_PATH = MODEL_DIR / "occ.pkl"
DEPTH_PATH = MODEL_DIR / "depth.pkl"
FEATURE_PATH = MODEL_DIR / "feat.pkl"
METRIC_PATH = MODEL_DIR / "metrics.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"
COORDS_PATH = MODEL_DIR / "coords.pkl"
