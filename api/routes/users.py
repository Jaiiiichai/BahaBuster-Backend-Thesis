
"""User-related endpoints backed by Supabase."""
from fastapi import APIRouter, HTTPException, Request
from fastapi import status
from pydantic import BaseModel
import bcrypt


from ..schemas import (
    UserCreateRequest,
    UserPushTokenUpsertRequest,
    UserPushTokenUpsertResponse,
    UserResponse,
)
from ..supabase_client import (
    fetch_users,
    insert_user,
    upsert_user_push_token,
)

router = APIRouter(tags=["users"])

# Schema for partial user update
class UserUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    password: str | None = None




@router.patch("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, payload: UserUpdateRequest, request: Request):
    """Update user details (name, email, password). Only provided fields are updated."""
    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    update_data = {}
    if payload.name is not None:
        update_data["name"] = payload.name
    if payload.email is not None:
        update_data["email"] = payload.email
    if payload.password is not None:
        update_data["password_hash"] = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update.")

    endpoint = f"{supabase_client.url}/rest/v1/users"
    params = {"user_id": f"eq.{user_id}"}
    headers = {"Prefer": "return=representation"}
    try:
        response = supabase_client.session.patch(endpoint, params=params, json=update_data, headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase user update failed: {exc}") from exc

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="User not found.")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase user update failed with HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise HTTPException(status_code=404, detail="User not found after update.")
    return payload[0]

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, request: Request):
    """Delete a user by their user_id."""
    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    endpoint = f"{supabase_client.url}/rest/v1/users"
    params = {"user_id": f"eq.{user_id}"}
    try:
        response = supabase_client.session.delete(endpoint, params=params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase user delete failed: {exc}") from exc

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="User not found.")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase user delete failed with HTTP {response.status_code}: {response.text}")
    # No content returned on success
    return


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


@router.post("/users/admin", response_model=UserResponse, status_code=201)
def create_admin_user(payload: UserCreateRequest, request: Request):
    """Insert one admin user row into Supabase."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        data = payload.model_dump()
        plain_password = data.pop("password")
        data["password_hash"] = bcrypt.hashpw(
            plain_password.encode(), bcrypt.gensalt()
        ).decode()
        data["role"] = "ADMIN"
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
