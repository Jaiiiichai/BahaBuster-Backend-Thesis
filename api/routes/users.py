"""User-related endpoints backed by Supabase."""
from fastapi import APIRouter, HTTPException, Request

import bcrypt

from ..schemas import (
    UserCreateRequest,
    UserPushTokenUpsertRequest,
    UserPushTokenUpsertResponse,
    UserResponse,
)
from ..supabase_client import fetch_users, insert_user, upsert_user_push_token

router = APIRouter(tags=["users"])


@router.get("/users", response_model=list[UserResponse])
def get_users(request: Request):
    """Return all users from Supabase."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return fetch_users(supabase_client)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/users", response_model=UserResponse, status_code=201)
def create_user(payload: UserCreateRequest, request: Request):
    """Insert one user row into Supabase."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        data = payload.model_dump()
        plain_password = data.pop("password")
        data["password_hash"] = bcrypt.hashpw(
            plain_password.encode(), bcrypt.gensalt()
        ).decode()
        data["role"] = "USER"
        return insert_user(supabase_client, data)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/users/push-token", response_model=UserPushTokenUpsertResponse)
def register_or_update_push_token(payload: UserPushTokenUpsertRequest, request: Request):
    """Insert user's push token or replace stored token if changed."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        return upsert_user_push_token(
            supabase_client,
            user_id=payload.user_id,
            expo_push_token=payload.expo_push_token,
            barangay=payload.barangay,
        )
    except RuntimeError as exc:
        detail = str(exc)
        if detail == "Invalid Expo push token format.":
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=502, detail=detail) from exc
