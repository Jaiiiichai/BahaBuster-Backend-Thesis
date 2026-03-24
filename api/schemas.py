"""Pydantic schemas shared across API routes."""
from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ManualPredictionRequest(BaseModel):
    """Payload contract for manually triggered prediction requests."""

    barangay: Optional[str] = None
    features: Dict[str, float]
    actual_flood: Optional[bool] = None
    actual_depth_cm: Optional[float] = None


class UserResponse(BaseModel):
    """Serialized user row returned by the users endpoint."""

    user_id: int
    email: str
    name: str
    barangay: str
    password_hash: str
    role: str
    created_at: str


class UserCreateRequest(BaseModel):
    """Payload contract for creating a user in Supabase."""

    email: str
    name: str
    barangay: str
    password: str


class LoginRequest(BaseModel):
    """Credentials for user login."""

    email: str
    password: str


class LoginResponse(BaseModel):
    """JWT token returned on successful login."""

    access_token: str
    token_type: str = "bearer"
    user_id: int
    name: str
    email: str
    barangay: str
    role: str


class FloodReportCreateRequest(BaseModel):
    """Payload contract for creating a flood report."""

    severity: str
    description: Optional[str] = None
    photos: list[str] = Field(default_factory=list)
    user_barangay: str
    user_email: str


class FloodReportResponse(BaseModel):
    """Serialized flood report row returned by the reports endpoints."""

    report_id: int
    severity: str
    description: Optional[str] = None
    photos: list[str]
    user_barangay: str
    user_email: str
    timestamp: datetime
    created_at: datetime


class AlertCreateRequest(BaseModel):
    """Payload contract for creating an alert."""

    title: str
    location: str
    description: Optional[str] = None
    severity: Literal["critical", "moderate", "low"]
    status: Literal["active", "resolved"] = "active"
    acknowledged: bool = False


class AlertResponse(BaseModel):
    """Serialized alert row returned by the alerts endpoints."""

    id: int
    title: str
    location: str
    description: Optional[str] = None
    severity: Literal["critical", "moderate", "low"]
    status: Literal["active", "resolved"]
    acknowledged: bool
    created_at: datetime


class ImageAnalysisResponse(BaseModel):
    """Response from image flood analysis."""

    is_flood: bool
    flood_classification: Literal["FLOOD", "NO FLOOD"]
    water_percentage: float
    description: str
    short_summary: str
    severity: Literal["mild", "moderate", "severe", "none"]
    confidence: float
    recommendations: list[str]
    image_info: Dict
    barangay: Optional[str] = None
