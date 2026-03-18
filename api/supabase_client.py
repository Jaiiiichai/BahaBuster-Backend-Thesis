"""Supabase client configuration and connectivity helpers."""
from __future__ import annotations

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
