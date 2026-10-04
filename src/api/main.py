"""FastAPI service: /health, /model/info, /predict, /explain."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from ..service import PredictionService
from ..utils import get_logger
from .schemas import ExplainResponse, HealthResponse, OrderRequest, PredictResponse

log = get_logger(__name__)


def create_app(model_dir: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        try:
            app.state.service = PredictionService.load(model_dir or os.environ.get("LATESHIP_MODEL_DIR"))
            log.info("Loaded model %s", app.state.service.version)
        except FileNotFoundError as exc:
            log.warning("Starting without a model: %s", exc)
        yield

    app = FastAPI(title="LateShip", description="Leakage-safe delivery risk intelligence", version="1.0.0",
                  lifespan=lifespan)  # fmt: skip

    def get_service(request: Request) -> PredictionService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(503, "Model not loaded. Train one with `make demo` or `make train`.")
        return svc

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        loaded = request.app.state.service is not None
        return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}

    @app.get("/model/info")
    def model_info(svc: PredictionService = Depends(get_service)):
        return {**svc.metadata, "threshold_info": svc.threshold_info}

    @app.post("/predict", response_model=PredictResponse)
    def predict(order: OrderRequest, svc: PredictionService = Depends(get_service)):
        return svc.predict([order.model_dump()])[0]

    @app.post("/explain", response_model=ExplainResponse)
    def explain(order: OrderRequest, top_k: int = Query(5, ge=1, le=20),
                svc: PredictionService = Depends(get_service)):  # fmt: skip
        return svc.explain(order.model_dump(), top_k)

    return app


app = create_app()
