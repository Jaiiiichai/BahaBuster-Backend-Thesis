"""Pydantic schemas shared across API routes."""
from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class UserPushTokenUpsertRequest(BaseModel):
    """Payload contract for registering/updating a user's Expo push token."""

    user_id: int
    barangay: Optional[str] = None
    expo_push_token: str


class UserPushTokenUpsertResponse(BaseModel):
    """Result returned after token registration/update checks."""

    action: Literal["inserted", "updated", "unchanged"]
    record: dict


class SosEventCreateRequest(BaseModel):
    """Payload contract for creating an SOS event and triggering alerts."""

    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    barangay: str
    latitude: float
    longitude: float
    timestamp: Optional[datetime] = None
    message: Optional[str] = None
    status: Literal["active", "resolved"] = "active"

    @field_validator("user_id", mode="before")
    @classmethod
    def parse_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                return int(digits)
        raise ValueError("userId must contain a numeric user id.")


class SosEventResponse(BaseModel):
    """Serialized SOS event row returned after creation."""

    sos_id: int
    user_id: int
    barangay: str
    latitude: float
    longitude: float
    message: Optional[str] = None
    status: Literal["active", "resolved"]
    expires_at: datetime
    created_at: datetime


class SosCreateAndNotifyResponse(BaseModel):
    """Combined SOS creation response with notification dispatch summary."""

    sos_event: SosEventResponse
    recipients_total: int
    notifications_sent: int
    notification_tickets: list[dict]


class SosMapEventResponse(BaseModel):
    """Active SOS event fields for map rendering, including requester name."""

    sos_id: int
    user_id: int
    requester_name: Optional[str] = None
    barangay: str
    latitude: float
    longitude: float
    message: Optional[str] = None
    status: Literal["active", "resolved"]
    expires_at: datetime
    created_at: datetime


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


class AlertAcknowledgeRequest(BaseModel):
    """Payload contract for updating the acknowledged field of an alert."""

    acknowledged: bool


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


class AutoAlertGenerateResponse(BaseModel):
    """Result returned after automatic alert generation from prediction + OpenAI."""

    title: str
    description: str


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
