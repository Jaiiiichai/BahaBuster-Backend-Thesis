"""Alert endpoints backed by Supabase."""
import os

import requests
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from model_training import ModelNotFoundError, predict_with_weather

from ..openai_client import OpenAIIntegrationError, generate_alert_copy_from_flood_data
from ..schemas import (
    AlertAcknowledgeRequest,
    AlertCreateRequest,
    AlertResponse,
    AutoAlertGenerateResponse,
)


from ..supabase_client import (
    insert_alert,
    fetch_push_tokens_by_barangay,
    fetch_flood_reports_by_barangay,
    fetch_alerts_by_barangay,
    update_alert_acknowledged,
)
from fastapi import status
router = APIRouter(tags=["alerts"])

@router.delete("/alerts/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_alert(alert_id: int, request: Request):
    """Delete an alert by its ID."""
    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    endpoint = f"{supabase_client.url}/rest/v1/alerts"
    params = {"id": f"eq.{alert_id}"}
    try:
        response = supabase_client.session.delete(endpoint, params=params)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Supabase alert delete failed: {exc}") from exc

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Alert not found.")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase alert delete failed with HTTP {response.status_code}: {response.text}")
    # No content returned on success
    return

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
DEFAULT_EXPO_PUSH_TOKEN = os.getenv("EXPO_PUSH_TOKEN", "")
MAX_EXPO_BATCH_SIZE = 1000
ALERT_NOTIFICATION_ACTIVE_HOURS = 4


class ExpoNotificationRequest(BaseModel):
    to: str = Field(default=DEFAULT_EXPO_PUSH_TOKEN, description="Expo push token")
    title: str = Field(default="Notification")
    body: str = Field(default="Message from backend")
    data: dict = Field(default_factory=dict)
    sound: str = Field(default="default")
    priority: str = Field(default="high")
    channelId: str | None = Field(default="default")


class ExpoReceiptRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, description="Expo ticket IDs")


class ExpoBatchNotificationRequest(BaseModel):
    tokens: list[str] = Field(..., min_length=1, description="List of Expo push tokens")
    title: str = Field(default="Notification")
    body: str = Field(default="Message from backend")
    data: dict = Field(default_factory=dict)
    sound: str = Field(default="default")
    priority: str = Field(default="high")
    channelId: str | None = Field(default="default")


def _expo_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Accept-encoding": "gzip, deflate",
        "Content-Type": "application/json",
    }


def _is_valid_expo_token(token: str) -> bool:
    return token.startswith("ExponentPushToken[")


def send_notification(payload: ExpoNotificationRequest) -> dict:
    if not payload.to:
        raise HTTPException(
            status_code=400,
            detail="Missing Expo push token. Provide 'to' in the request body or set EXPO_PUSH_TOKEN.",
        )

    if not payload.to.startswith("ExponentPushToken["):
        raise HTTPException(status_code=400, detail="Invalid Expo push token format.")

    headers = _expo_headers()

    expo_payload = {
        "to": payload.to,
        "title": payload.title,
        "body": payload.body,
        "data": payload.data,
        "sound": payload.sound,
        "priority": payload.priority,
        "channelId": payload.channelId,
    }

    try:
        response = requests.post(EXPO_PUSH_URL, json=expo_payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send Expo notification: {exc}") from exc


def get_expo_receipts(payload: ExpoReceiptRequest) -> dict:
    headers = _expo_headers()

    try:
        response = requests.post(EXPO_RECEIPTS_URL, json={"ids": payload.ids}, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Expo receipts: {exc}") from exc


def send_batch_notifications(payload: ExpoBatchNotificationRequest) -> dict:
    normalized_tokens = list(dict.fromkeys(token.strip() for token in payload.tokens if token and token.strip()))
    if not normalized_tokens:
        raise HTTPException(status_code=400, detail="No valid tokens provided.")

    invalid_tokens = [token for token in normalized_tokens if not _is_valid_expo_token(token)]
    if invalid_tokens:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "One or more tokens are not valid Expo push tokens.",
                "invalid_count": len(invalid_tokens),
                "invalid_tokens": invalid_tokens,
            },
        )

    messages = [
        {
            "to": token,
            "title": payload.title,
            "body": payload.body,
            "data": payload.data,
            "sound": payload.sound,
            "priority": payload.priority,
            "channelId": payload.channelId,
        }
        for token in normalized_tokens
    ]

    headers = _expo_headers()
    all_tickets: list[dict] = []
    chunks = [messages[i : i + MAX_EXPO_BATCH_SIZE] for i in range(0, len(messages), MAX_EXPO_BATCH_SIZE)]

    try:
        for chunk in chunks:
            response = requests.post(EXPO_PUSH_URL, json=chunk, headers=headers, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            ticket_data = response_json.get("data", [])
            if isinstance(ticket_data, list):
                all_tickets.extend(ticket_data)
            else:
                all_tickets.append(ticket_data)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send batch Expo notifications: {exc}") from exc

    return {
        "token_count": len(normalized_tokens),
        "chunk_count": len(chunks),
        "expo_response": {"data": all_tickets},
    }


@router.post("/alerts", response_model=AlertResponse, status_code=201)
def create_alert(payload: AlertCreateRequest, request: Request):
    """Create an alert in Supabase."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        created_alert = insert_alert(supabase_client, payload.model_dump(mode="json"))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        recipient_tokens = fetch_push_tokens_by_barangay(
            supabase_client,
            barangay=created_alert.get("location", payload.location),
            active_within_hours=ALERT_NOTIFICATION_ACTIVE_HOURS,
        )
        if recipient_tokens:
            alert_title = (created_alert.get("title") or "Barangay Alert").strip()
            alert_body = (
                (created_alert.get("description") or payload.description or created_alert.get("title") or "")
                .strip()
            )
            if not alert_body:
                alert_body = f"New alert in {created_alert.get('location', payload.location)}."

            send_batch_notifications(
                ExpoBatchNotificationRequest(
                    tokens=recipient_tokens,
                    title=alert_title,
                    body=alert_body,
                    data={
                        "type": "alert",
                        "alert_id": created_alert.get("id"),
                        "location": created_alert.get("location", payload.location),
                        "severity": created_alert.get("severity"),
                        "status": created_alert.get("status"),
                    },
                )
            )
    except (RuntimeError, HTTPException) as exc:
        print(f"Alert notification dispatch failed: {exc}")

    return created_alert


@router.get("/alerts/auto-generate/{barangay}", response_model=AutoAlertGenerateResponse)
def auto_generate_alert(barangay: str, request: Request):
    """Generate alert title and description from barangay flood prediction/report data."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        prediction = predict_with_weather(barangay)
    except ModelNotFoundError as exc:
        available = ", ".join(exc.available) or "none"
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for barangay '{barangay}'. Available: {available}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch prediction data: {exc}") from exc

    try:
        recent_reports = fetch_flood_reports_by_barangay(supabase_client, barangay, timeout=10)
    except RuntimeError:
        recent_reports = []

    generated = None
    try:
        generated = generate_alert_copy_from_flood_data(
            barangay=barangay,
            prediction=prediction,
            recent_reports=recent_reports,
        )
    except OpenAIIntegrationError:
        day1 = (prediction.get("predictions") or [{}])[0]
        generated = {
            "title": f"Flood Alert: {barangay}",
            "description": (
                f"Forecast indicates {day1.get('risk_level', 'LOW')} flood risk in {barangay}. "
                f"Expected depth is around {day1.get('predicted_depth_cm', 0)} cm. Stay alert and monitor updates."
            ),
            "severity": "moderate",
            "reason": "OpenAI unavailable, used prediction-based fallback copy.",
        }

    return {
        "title": generated["title"],
        "description": generated["description"],
    }


@router.get("/alerts", response_model=list[AlertResponse])
def get_alerts_by_barangay(
    request: Request,
    barangay: str = Query(..., description="Barangay name used to filter alerts case-insensitively"),
):
    """Return alerts filtered by barangay/location case-insensitively."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return fetch_alerts_by_barangay(supabase_client, barangay)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.patch("/alerts/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(alert_id: int, payload: AlertAcknowledgeRequest, request: Request):
    """Update the acknowledged field of a specific alert."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return update_alert_acknowledged(supabase_client, alert_id, payload.acknowledged)
    except RuntimeError as exc:
        msg = str(exc)
        status = 404 if "not found" in msg.lower() else 502
        raise HTTPException(status_code=status, detail=msg) from exc


@router.post("/send")
def trigger_notification(payload: ExpoNotificationRequest):
    result = send_notification(payload)
    return {"message": "Notification sent", "expo_response": result}


@router.post("/send/receipt")
def get_notification_receipts(payload: ExpoReceiptRequest):
    result = get_expo_receipts(payload)
    return {"message": "Receipt fetched", "expo_response": result}


@router.post("/send/batch")
def trigger_batch_notification(payload: ExpoBatchNotificationRequest):
    result = send_batch_notifications(payload)
    return {"message": "Batch notification request processed", **result}
