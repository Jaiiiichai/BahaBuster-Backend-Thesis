"""Pydantic schemas shared across API routes."""
from typing import Dict, Optional

from pydantic import BaseModel


class ManualPredictionRequest(BaseModel):
    """Payload contract for manually triggered prediction requests."""

    barangay: Optional[str] = None
    features: Dict[str, float]
    actual_flood: Optional[bool] = None
    actual_depth_cm: Optional[float] = None
