"""SOS endpoints for emergency events and push fanout."""
from datetime import datetime, timedelta, timezone

import requests
from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import SosCreateAndNotifyResponse, SosEventCreateRequest, SosMapEventResponse
from ..supabase_client import fetch_active_push_tokens_for_sos, fetch_active_sos_events, insert_sos_event

router = APIRouter(tags=["sos"])

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
MAX_EXPO_BATCH_SIZE = 100
TOKEN_ACTIVE_WINDOW_HOURS = 4


def _expo_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Accept-encoding": "gzip, deflate",
        "Content-Type": "application/json",
    }


def _send_batch_to_expo(tokens: list[str], title: str, body: str, data: dict) -> list[dict]:
    if not tokens:
        return []

    messages = [
        {
            "to": token,
            "title": title,
            "body": body,
            "data": data,
            "sound": "default",
            "priority": "high",
            "channelId": "default",
        }
        for token in tokens
    ]

    chunks = [messages[i : i + MAX_EXPO_BATCH_SIZE] for i in range(0, len(messages), MAX_EXPO_BATCH_SIZE)]
    tickets: list[dict] = []

    for chunk in chunks:
        try:
            response = requests.post(EXPO_PUSH_URL, json=chunk, headers=_expo_headers(), timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"Failed to send SOS notifications: {exc}") from exc

        payload = response.json()
        ticket_data = payload.get("data", [])
        if isinstance(ticket_data, list):
            tickets.extend(ticket_data)
        else:
            tickets.append(ticket_data)

    return tickets


@router.post("/sos", response_model=SosCreateAndNotifyResponse, status_code=201)
def create_sos_event(payload: SosEventCreateRequest, request: Request):
    """Create an SOS event, then notify all active token holders, including the sender."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    created_at = payload.timestamp or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    expires_at = created_at + timedelta(hours=TOKEN_ACTIVE_WINDOW_HOURS)

    sos_row = {
        "user_id": payload.user_id,
        "barangay": payload.barangay,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "message": payload.message,
        "status": payload.status,
        "expires_at": expires_at.isoformat(),
    }

    try:
        sos_event = insert_sos_event(supabase_client, sos_row)
        recipient_tokens = fetch_active_push_tokens_for_sos(
            supabase_client,
            active_within_hours=TOKEN_ACTIVE_WINDOW_HOURS,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    notif_title = "SOS Alert"
    notif_body = f"Emergency reported in {payload.barangay}."
    if payload.message:
        notif_body = payload.message

    tickets = _send_batch_to_expo(
        recipient_tokens,
        title=notif_title,
        body=notif_body,
        data={
            "type": "sos",
            "sos_id": sos_event.get("sos_id"),
            "barangay": payload.barangay,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "status": payload.status,
            "expires_at": sos_event.get("expires_at"),
        },
    )

    notifications_sent = sum(1 for ticket in tickets if ticket.get("status") == "ok")

    return {
        "sos_event": sos_event,
        "recipients_total": len(recipient_tokens),
        "notifications_sent": notifications_sent,
        "notification_tickets": tickets,
    }


@router.get("/sos", response_model=list[SosMapEventResponse])
def get_active_sos_events(
    request: Request,
    barangay: str | None = Query(default=None, description="Optional barangay filter"),
):
    """Return active, non-expired SOS points for map display with requester name."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return fetch_active_sos_events(supabase_client, barangay=barangay)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
