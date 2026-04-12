"""Authentication endpoints."""
import logging
import os
import time

import bcrypt
from fastapi import APIRouter, HTTPException, Request

from ..schemas import LoginRequest, LoginResponse
from ..security import create_access_token
from ..supabase_client import fetch_user_by_email, update_user_password_hash

router = APIRouter(tags=["auth"])
logger = logging.getLogger(__name__)

LOGIN_DB_TIMEOUT_SECONDS = int(os.getenv("LOGIN_DB_TIMEOUT_SECONDS", "5"))
UPGRADE_LEGACY_HASH_ON_LOGIN = os.getenv("UPGRADE_LEGACY_HASH_ON_LOGIN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request):
    """Validate credentials and return a JWT access token."""

    login_started = time.perf_counter()

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        db_started = time.perf_counter()
        user = fetch_user_by_email(supabase_client, payload.email, timeout=LOGIN_DB_TIMEOUT_SECONDS)
        db_ms = (time.perf_counter() - db_started) * 1000
    except RuntimeError as exc:
        logger.warning("login_failed_db email=%s error=%s", payload.email, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    stored_hash: str = user["password_hash"]
    password_valid = False

    if stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            bcrypt_started = time.perf_counter()
            password_valid = bcrypt.checkpw(payload.password.encode(), stored_hash.encode())
            bcrypt_ms = (time.perf_counter() - bcrypt_started) * 1000
        except ValueError:
            password_valid = False
            bcrypt_ms = 0.0
    elif payload.password == stored_hash:
        password_valid = True
        bcrypt_ms = 0.0
        if UPGRADE_LEGACY_HASH_ON_LOGIN:
            try:
                upgrade_started = time.perf_counter()
                upgraded_hash = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()
                user = update_user_password_hash(
                    supabase_client,
                    user_id=user["user_id"],
                    password_hash=upgraded_hash,
                    timeout=LOGIN_DB_TIMEOUT_SECONDS,
                )
                upgrade_ms = (time.perf_counter() - upgrade_started) * 1000
                logger.info("login_legacy_hash_upgraded user_id=%s upgrade_ms=%.2f", user["user_id"], upgrade_ms)
            except RuntimeError as exc:
                logger.warning("login_legacy_hash_upgrade_failed user_id=%s error=%s", user.get("user_id"), str(exc))

    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
    )

    total_ms = (time.perf_counter() - login_started) * 1000
    logger.info(
        "login_success user_id=%s email=%s db_ms=%.2f bcrypt_ms=%.2f total_ms=%.2f",
        user["user_id"],
        user["email"],
        db_ms,
        bcrypt_ms,
        total_ms,
    )

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user_id=user["user_id"],
        name=user["name"],
        email=user["email"],
        barangay=user["barangay"],
        role=user["role"],
    )
