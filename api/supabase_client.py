"""Supabase client configuration and connectivity helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from dataclasses import dataclass
from typing import Tuple

import requests
from dotenv import load_dotenv


class SupabaseConfigError(RuntimeError):
    """Raised when required Supabase configuration values are missing."""


@dataclass
class SupabaseClient:
    """Minimal HTTP client wrapper for Supabase REST access."""

    url: str
    key: str
    session: requests.Session


def _load_credentials() -> tuple[str, str]:
    """Load Supabase credentials from environment variables."""

    load_dotenv()
    url = os.getenv("SUPABASE_URL", "").strip()
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    publishable_key = os.getenv("SUPABASE_KEY", "").strip()
    key = service_role_key or publishable_key

    if not url or not key:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in environment variables."
        )

    return url, key


def create_supabase_client() -> SupabaseClient:
    """Return a configured Supabase REST client wrapper."""

    url, key = _load_credentials()
    session = requests.Session()
    session.headers.update(
        {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
    )
    return SupabaseClient(url=url.rstrip("/"), key=key, session=session)


def verify_supabase_connection(timeout: int = 10) -> Tuple[bool, str]:
    """Check Supabase key validity against the Auth settings endpoint."""

    url, key = _load_credentials()
    endpoint = f"{url.rstrip('/')}/auth/v1/settings"
    headers = {
        "apikey": key,
    }

    try:
        response = requests.get(endpoint, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"Connection error: {exc}"

    if response.ok:
        return True, "Supabase connection established."

    return False, f"Supabase returned HTTP {response.status_code}."


def fetch_users(client: SupabaseClient, timeout: int = 10) -> list[dict]:
    """Fetch rows from the users table via Supabase REST."""

    endpoint = f"{client.url}/rest/v1/users"
    params = {
        "select": "user_id,email,name,barangay,password_hash,role,created_at",
        "order": "user_id.asc",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase request failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase query failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Supabase response format for users table.")

    return payload


def fetch_user_by_email(client: SupabaseClient, email: str, timeout: int = 10) -> dict | None:
    """Return a single user row matching the given email, or None."""

    endpoint = f"{client.url}/rest/v1/users"
    params = {
        "select": "user_id,email,name,barangay,password_hash,role,created_at",
        "email": f"eq.{email}",
        "limit": "1",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase request failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase query failed with HTTP {response.status_code}: {response.text}")

    rows = response.json()
    return rows[0] if rows else None


def insert_user(client: SupabaseClient, user: dict, timeout: int = 10) -> dict:
    """Insert one user row into the users table and return created record."""

    endpoint = f"{client.url}/rest/v1/users"
    headers = {
        "Prefer": "return=representation",
    }

    try:
        response = client.session.post(endpoint, json=user, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase insert failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase insert failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Supabase insert did not return the created user.")

    return payload[0]


def update_user_password_hash(
    client: SupabaseClient,
    user_id: int,
    password_hash: str,
    timeout: int = 10,
) -> dict:
    """Update a user's stored password hash and return the updated record."""

    endpoint = f"{client.url}/rest/v1/users"
    params = {
        "user_id": f"eq.{user_id}",
    }
    headers = {
        "Prefer": "return=representation",
    }

    try:
        response = client.session.patch(
            endpoint,
            params=params,
            json={"password_hash": password_hash},
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase password update failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase password update failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Supabase password update did not return the updated user.")

    return payload[0]


def insert_flood_report(client: SupabaseClient, report: dict, timeout: int = 10) -> dict:
    """Insert one flood report row and return the created record."""

    endpoint = f"{client.url}/rest/v1/flood_reports"
    headers = {
        "Prefer": "return=representation",
    }

    try:
        response = client.session.post(endpoint, json=report, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase flood report insert failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase flood report insert failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Supabase flood report insert did not return the created report.")

    return payload[0]


def fetch_flood_reports_by_barangay(
    client: SupabaseClient,
    barangay: str,
    timeout: int = 10,
) -> list[dict]:
    """Fetch flood reports filtered by barangay name using case-insensitive matching."""

    endpoint = f"{client.url}/rest/v1/flood_reports"
    params = {
        "select": "report_id,severity,description,photos,user_barangay,user_email,timestamp,created_at",
        "user_barangay": f"ilike.{barangay.strip()}",
        "order": "timestamp.desc",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase flood report query failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase flood report query failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Supabase response format for flood reports.")

    return payload


def insert_alert(client: SupabaseClient, alert: dict, timeout: int = 10) -> dict:
    """Insert one alert row and return the created record."""

    endpoint = f"{client.url}/rest/v1/alerts"
    headers = {
        "Prefer": "return=representation",
    }

    try:
        response = client.session.post(endpoint, json=alert, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase alert insert failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase alert insert failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Supabase alert insert did not return the created alert.")

    return payload[0]


def update_alert_acknowledged(client: SupabaseClient, alert_id: int, acknowledged: bool, timeout: int = 10) -> dict:
    """Update the acknowledged field of a specific alert row and return the updated record."""

    endpoint = f"{client.url}/rest/v1/alerts"
    headers = {
        "Prefer": "return=representation",
    }
    params = {"id": f"eq.{alert_id}"}

    try:
        response = client.session.patch(
            endpoint, json={"acknowledged": acknowledged}, headers=headers, params=params, timeout=timeout
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase alert update failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase alert update failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Alert with id {alert_id} not found.")

    return payload[0]


def fetch_alerts_by_barangay(
    client: SupabaseClient,
    barangay: str,
    timeout: int = 10,
) -> list[dict]:
    """Fetch alerts filtered by barangay/location using case-insensitive matching."""

    endpoint = f"{client.url}/rest/v1/alerts"
    params = {
        "select": "id,title,location,description,severity,status,acknowledged,created_at",
        "location": f"ilike.{barangay.strip()}",
        "order": "created_at.desc",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase alerts query failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase alerts query failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Supabase response format for alerts.")

    return payload


def upsert_user_push_token(
    client: SupabaseClient,
    user_id: int,
    expo_push_token: str,
    barangay: str | None = None,
    timeout: int = 10,
) -> dict:
    """Insert or update a user's Expo push token based on current stored value."""

    if not expo_push_token.startswith("ExponentPushToken["):
        raise RuntimeError("Invalid Expo push token format.")

    endpoint = f"{client.url}/rest/v1/user_push_tokens"
    select_params = {
        "select": "id,user_id,barangay,expo_push_token,created_at",
        "user_id": f"eq.{user_id}",
        "order": "id.desc",
        "limit": "1",
    }

    try:
        existing_response = client.session.get(endpoint, params=select_params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase token query failed: {exc}") from exc

    if existing_response.status_code >= 400:
        raise RuntimeError(
            f"Supabase token query failed with HTTP {existing_response.status_code}: {existing_response.text}"
        )

    existing_rows = existing_response.json()
    latest_row = existing_rows[0] if existing_rows else None

    def _cleanup_old_rows(keep_id: int) -> None:
        delete_params = {
            "user_id": f"eq.{user_id}",
            "id": f"neq.{keep_id}",
        }
        try:
            cleanup_response = client.session.delete(endpoint, params=delete_params, timeout=timeout)
        except requests.RequestException as exc:
            raise RuntimeError(f"Supabase token cleanup failed: {exc}") from exc

        if cleanup_response.status_code >= 400:
            raise RuntimeError(
                f"Supabase token cleanup failed with HTTP {cleanup_response.status_code}: {cleanup_response.text}"
            )

    headers = {
        "Prefer": "return=representation",
    }

    if latest_row and latest_row.get("expo_push_token") == expo_push_token:
        update_params = {
            "id": f"eq.{latest_row['id']}",
        }
        payload = {
            # Re-touch created_at so token can be treated as active during the next 4-hour window.
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if barangay is not None:
            payload["barangay"] = barangay

        try:
            touch_response = client.session.patch(
                endpoint,
                params=update_params,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Supabase token refresh failed: {exc}") from exc

        if touch_response.status_code >= 400:
            raise RuntimeError(
                f"Supabase token refresh failed with HTTP {touch_response.status_code}: {touch_response.text}"
            )

        touched_rows = touch_response.json()
        if not isinstance(touched_rows, list) or not touched_rows:
            raise RuntimeError("Supabase token refresh did not return updated rows.")

        _cleanup_old_rows(keep_id=touched_rows[0]["id"])

        return {
            "action": "unchanged",
            "record": touched_rows[0],
        }

    if latest_row:
        update_params = {
            "id": f"eq.{latest_row['id']}",
        }
        payload = {
            "expo_push_token": expo_push_token,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if barangay is not None:
            payload["barangay"] = barangay

        try:
            update_response = client.session.patch(
                endpoint,
                params=update_params,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Supabase token update failed: {exc}") from exc

        if update_response.status_code >= 400:
            raise RuntimeError(
                f"Supabase token update failed with HTTP {update_response.status_code}: {update_response.text}"
            )

        updated_rows = update_response.json()
        if not isinstance(updated_rows, list) or not updated_rows:
            raise RuntimeError("Supabase token update did not return updated rows.")

        _cleanup_old_rows(keep_id=updated_rows[0]["id"])

        return {
            "action": "updated",
            "record": updated_rows[0],
        }

    insert_payload = {
        "user_id": user_id,
        "expo_push_token": expo_push_token,
    }
    if barangay is not None:
        insert_payload["barangay"] = barangay

    try:
        insert_response = client.session.post(
            endpoint,
            json=insert_payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase token insert failed: {exc}") from exc

    if insert_response.status_code >= 400:
        raise RuntimeError(
            f"Supabase token insert failed with HTTP {insert_response.status_code}: {insert_response.text}"
        )

    inserted_rows = insert_response.json()
    if not isinstance(inserted_rows, list) or not inserted_rows:
        raise RuntimeError("Supabase token insert did not return the created row.")

    _cleanup_old_rows(keep_id=inserted_rows[0]["id"])

    return {
        "action": "inserted",
        "record": inserted_rows[0],
    }


def insert_sos_event(client: SupabaseClient, event: dict, timeout: int = 10) -> dict:
    """Insert one SOS event row and return the created record."""

    endpoint = f"{client.url}/rest/v1/sos_events"
    headers = {
        "Prefer": "return=representation",
    }

    try:
        response = client.session.post(endpoint, json=event, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase SOS insert failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase SOS insert failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Supabase SOS insert did not return the created event.")

    return payload[0]


def fetch_active_push_tokens_for_sos(
    client: SupabaseClient,
    active_within_hours: int = 4,
    exclude_user_id: int | None = None,
    timeout: int = 10,
) -> list[str]:
    """Fetch active Expo push tokens for SOS fanout, excluding the SOS sender."""

    endpoint = f"{client.url}/rest/v1/user_push_tokens"
    active_since = (datetime.now(timezone.utc) - timedelta(hours=active_within_hours)).isoformat()
    params = {
        "select": "expo_push_token",
        "created_at": f"gte.{active_since}",
        "order": "id.desc",
    }
    if exclude_user_id is not None:
        params["user_id"] = f"neq.{exclude_user_id}"

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase push-token query failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase push-token query failed with HTTP {response.status_code}: {response.text}")

    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected Supabase response format for push tokens.")

    tokens = [row.get("expo_push_token", "") for row in rows]
    valid = [token for token in tokens if isinstance(token, str) and token.startswith("ExponentPushToken[")]
    return list(dict.fromkeys(valid))


def fetch_push_tokens_by_barangay(
    client: SupabaseClient,
    barangay: str,
    active_within_hours: int = 4,
    timeout: int = 10,
) -> list[str]:
    """Fetch active Expo push tokens in a barangay for general alert broadcasts."""

    endpoint = f"{client.url}/rest/v1/user_push_tokens"
    active_since = (datetime.now(timezone.utc) - timedelta(hours=active_within_hours)).isoformat()
    params = {
        "select": "expo_push_token",
        "barangay": f"ilike.{barangay.strip()}",
        "created_at": f"gte.{active_since}",
        "order": "id.desc",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase push-token query failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase push-token query failed with HTTP {response.status_code}: {response.text}")

    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected Supabase response format for push tokens.")

    tokens = [row.get("expo_push_token", "") for row in rows]
    valid = [token for token in tokens if isinstance(token, str) and token.startswith("ExponentPushToken[")]
    return list(dict.fromkeys(valid))


def fetch_active_sos_by_user(client: SupabaseClient, user_id: int, timeout: int = 10) -> dict | None:
    """Return the first active, non-expired SOS event for the given user, or None."""

    endpoint = f"{client.url}/rest/v1/sos_events"
    now_iso = datetime.now(timezone.utc).isoformat()
    params = {
        "select": "sos_id,expires_at",
        "user_id": f"eq.{user_id}",
        "status": "eq.active",
        "expires_at": f"gte.{now_iso}",
        "order": "created_at.desc",
        "limit": "1",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase SOS duplicate check failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase SOS duplicate check failed with HTTP {response.status_code}: {response.text}")

    rows = response.json()
    return rows[0] if rows else None


def expire_elapsed_sos_events(client: SupabaseClient, timeout: int = 10) -> None:
    """Mark SOS events as expired when their expiration timestamp has passed."""

    endpoint = f"{client.url}/rest/v1/sos_events"
    now_iso = datetime.now(timezone.utc).isoformat()
    params = {
        "status": "eq.active",
        "expires_at": f"lt.{now_iso}",
    }

    try:
        response = client.session.patch(endpoint, params=params, json={"status": "expired"}, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase SOS expiry update failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase SOS expiry update failed with HTTP {response.status_code}: {response.text}")


def fetch_active_sos_events(client: SupabaseClient, timeout: int = 10) -> list[dict]:
    """Fetch active, non-expired SOS events and include requester name for map use."""

    expire_elapsed_sos_events(client, timeout=timeout)

    endpoint = f"{client.url}/rest/v1/sos_events"
    now_iso = datetime.now(timezone.utc).isoformat()
    params = {
        "select": "sos_id,user_id,barangay,latitude,longitude,message,status,expires_at,created_at,users(name)",
        "status": "eq.active",
        "expires_at": f"gte.{now_iso}",
        "order": "created_at.desc",
    }

    try:
        response = client.session.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase SOS query failed: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Supabase SOS query failed with HTTP {response.status_code}: {response.text}")

    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected Supabase response format for SOS events.")

    normalized: list[dict] = []
    for row in rows:
        user_rel = row.get("users")
        requester_name = None
        if isinstance(user_rel, dict):
            requester_name = user_rel.get("name")
        elif isinstance(user_rel, list) and user_rel:
            first = user_rel[0]
            if isinstance(first, dict):
                requester_name = first.get("name")

        normalized.append(
            {
                "sos_id": row.get("sos_id"),
                "user_id": row.get("user_id"),
                "barangay": row.get("barangay"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "message": row.get("message"),
                "status": row.get("status"),
                "expires_at": row.get("expires_at"),
                "created_at": row.get("created_at"),
                "requester_name": requester_name,
            }
        )

    return normalized


def upload_image_to_bucket(
    client: SupabaseClient,
    bucket_name: str,
    file_path: str,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
    timeout: int = 30,
) -> dict:
    """
    Upload an image to a Supabase storage bucket.
    
    Args:
        client: SupabaseClient instance
        bucket_name: Name of the storage bucket (e.g., 'flood-images')
        file_path: Path where the file will be stored in the bucket (e.g., 'banilad/2024-03-24-123456.jpg')
        image_bytes: Raw image bytes to upload
        content_type: MIME type of the image
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with upload details including the public URL
        
    Raises:
        RuntimeError: If upload fails
    """
    endpoint = f"{client.url}/storage/v1/object/{bucket_name}/{file_path}"
    headers = client.session.headers.copy()
    headers["Content-Type"] = content_type
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            data=image_bytes,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Supabase image upload failed: {exc}") from exc
    
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase image upload failed with HTTP {response.status_code}: {response.text}"
        )
    
    # Construct the public URL
    public_url = f"{client.url}/storage/v1/object/public/{bucket_name}/{file_path}"
    
    return {
        "file_path": file_path,
        "bucket_name": bucket_name,
        "public_url": public_url,
        "size_bytes": len(image_bytes),
    }
