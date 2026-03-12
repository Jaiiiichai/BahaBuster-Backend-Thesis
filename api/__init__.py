"""Application factory and router wiring for the Flood Prediction API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health, predictions


def create_app() -> FastAPI:
    """Instantiate the FastAPI application with middleware and routers."""

    app = FastAPI(title="Flood Prediction System")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(predictions.router)
    app.include_router(health.router)

    return app
