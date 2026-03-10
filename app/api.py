from fastapi import FastAPI

from .dependencies import get_model_registry
from .routers import prediction, system

app = FastAPI(title="BahaBuster API", version="1.0.0")


@app.on_event("startup")
def preload_resources():
    """Warm up shared resources so first request stays fast."""
    get_model_registry()


app.include_router(system.router)
app.include_router(prediction.router)
