"""Lazy-loading registry that keeps barangay models available in memory."""
from __future__ import annotations

from typing import Optional

import joblib

from .config import MODEL_DIR
from .exceptions import ModelNotFoundError
from .naming import normalize_barangay_name
from .training import train_model


def _load_cached_models() -> dict[str, dict]:
    """Return any cached barangay bundles stored on disk."""

    registry: dict[str, dict] = {}
    if not MODEL_DIR.exists():
        return registry

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
            normalized_name = normalize_barangay_name(barangay_name)
        except ValueError:
            continue

        bundle["barangay"] = normalized_name
        registry[normalized_name] = bundle

    return registry


def load_model_registry() -> dict[str, dict]:
    """Load models from disk, falling back to training only when necessary."""

    registry = _load_cached_models()
    if registry:
        return registry

    print("[LOAD] No cached models found. Training from scratch...")
    return train_model()


MODEL_REGISTRY = load_model_registry()


def get_model_registry() -> dict[str, dict]:
    """Return the current in-memory registry."""

    return MODEL_REGISTRY


def refresh_model_registry(force_retrain: bool = False) -> dict[str, dict]:
    """Reload the registry, forcing a retrain when requested."""

    global MODEL_REGISTRY
    if force_retrain:
        MODEL_REGISTRY = train_model(force_retrain=True)
    else:
        cached = _load_cached_models()
        MODEL_REGISTRY = cached if cached else train_model()
    return MODEL_REGISTRY


def get_available_barangays(registry: Optional[dict[str, dict]] = None) -> list[str]:
    """Return a sorted list of barangays that have a trained model."""

    registry = registry or MODEL_REGISTRY
    return sorted(registry.keys())


def get_model_for_barangay(barangay: str, registry: Optional[dict[str, dict]] = None) -> dict:
    """Fetch a barangay bundle or raise ModelNotFoundError if missing."""

    registry = registry or MODEL_REGISTRY
    normalized = normalize_barangay_name(barangay)
    try:
        return registry[normalized]
    except KeyError:
        raise ModelNotFoundError(normalized, get_available_barangays(registry))
