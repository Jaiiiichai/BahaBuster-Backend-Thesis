"""Pydantic schemas shared across API routes."""
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel


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
    photos: list[str] = []
    user_barangay: str
    user_email: str
    timestamp: datetime


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
