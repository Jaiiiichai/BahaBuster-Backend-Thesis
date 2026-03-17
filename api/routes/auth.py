"""Authentication endpoints."""
import bcrypt
from fastapi import APIRouter, HTTPException, Request

from ..schemas import LoginRequest, LoginResponse
from ..security import create_access_token
from ..supabase_client import fetch_user_by_email, update_user_password_hash

router = APIRouter(tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request):
    """Validate credentials and return a JWT access token."""

    supabase_client = getattr(request.app.state, "supabase", None)
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        user = fetch_user_by_email(supabase_client, payload.email)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    stored_hash: str = user["password_hash"]
    password_valid = False

    if stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            password_valid = bcrypt.checkpw(payload.password.encode(), stored_hash.encode())
        except ValueError:
            password_valid = False
    elif payload.password == stored_hash:
        password_valid = True
        try:
            upgraded_hash = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()
            user = update_user_password_hash(
                supabase_client,
                user_id=user["user_id"],
                password_hash=upgraded_hash,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
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
