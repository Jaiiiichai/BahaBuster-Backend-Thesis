import math
import numpy as np


def clean(value):
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return round(numeric, 4)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def get_risk_level(prob_percent: float) -> str:
    if prob_percent < 20:
        return "Low"
    if prob_percent < 50:
        return "Moderate"
    if prob_percent < 75:
        return "High"
    return "Severe"
