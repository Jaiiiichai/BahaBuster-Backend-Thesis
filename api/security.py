"""JWT token creation and verification utilities."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

load_dotenv()

_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_HOURS = 24


def _get_secret_key() -> str:
    """Resolve the JWT signing secret from the environment."""

    secret_key = os.getenv("JWT_SECRET_KEY", "").strip() or os.getenv("SECRET_KEY", "").strip()
    if not secret_key:
        return "change-this-secret-in-production"

    return secret_key


def create_access_token(user_id: int, email: str, role: str) -> str:
    """Return a signed JWT containing user identity claims."""

    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, _get_secret_key(), algorithm=_ALGORITHM)
