"""Application factory and router wiring for the Flood Prediction API."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import alerts, auth, health, predictions, reports, users, image_analysis, sos
from .supabase_client import SupabaseConfigError, create_supabase_client, verify_supabase_connection


def create_app() -> FastAPI:
    """Instantiate the FastAPI application with middleware and routers."""

    app = FastAPI(title="Flood Prediction System")
    verify_on_startup = os.getenv("SUPABASE_VERIFY_ON_STARTUP", "0").strip().lower() in {"1", "true", "yes"}

    try:
        supabase_client = create_supabase_client()
        if verify_on_startup:
            is_connected, message = verify_supabase_connection(timeout=3)
        else:
            is_connected, message = True, "Startup verification skipped (SUPABASE_VERIFY_ON_STARTUP=0)."
        app.state.supabase = supabase_client
        app.state.supabase_status = {
            "configured": True,
            "connected": is_connected,
            "message": message,
        }
    except SupabaseConfigError as exc:
        app.state.supabase = None
        app.state.supabase_status = {
            "configured": False,
            "connected": False,
            "message": str(exc),
        }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predictions.router)
    app.include_router(users.router)
    app.include_router(auth.router)
    app.include_router(alerts.router)
    app.include_router(sos.router)
    app.include_router(reports.router)
    app.include_router(health.router)
    app.include_router(image_analysis.router)

    return app
