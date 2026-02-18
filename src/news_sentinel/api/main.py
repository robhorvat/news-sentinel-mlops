from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException

from news_sentinel.inference.predictors import PredictorManager
from news_sentinel.inference.schemas import HealthResponse, PredictRequest, PredictResponse
from news_sentinel.observability.metrics import (
    ERRORS_TOTAL,
    PREDICTION_CONFIDENCE,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    metrics_response,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor_manager = PredictorManager.from_artifacts()
    yield


app = FastAPI(title="News Sentinel API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def capture_request_metrics(request, call_next):
    start = perf_counter()
    path = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception as exc:
        ERRORS_TOTAL.labels(path=path, error_type=type(exc).__name__).inc()
        status = "500"
        raise
    finally:
        elapsed = perf_counter() - start
        REQUESTS_TOTAL.labels(path=path, method=method, status=status).inc()
        REQUEST_LATENCY_SECONDS.labels(path=path, method=method).observe(elapsed)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    manager: PredictorManager = app.state.predictor_manager
    available = manager.available_models()
    status = "ok" if available else "degraded"
    return HealthResponse(status=status, available_models=available)


@app.get("/models")
def list_models() -> dict:
    manager: PredictorManager = app.state.predictor_manager
    return {"available_models": manager.available_models()}


@app.get("/")
def root() -> dict:
    return {
        "service": "news-sentinel-api",
        "status": "ok",
        "docs": "/docs",
        "health": "/healthz",
        "predict": "/predict",
        "metrics": "/metrics",
    }


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    manager: PredictorManager = app.state.predictor_manager

    try:
        result = manager.predict(text=payload.text, requested_model=payload.model)
    except ValueError as exc:
        ERRORS_TOTAL.labels(path="/predict", error_type="ValueError").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    PREDICTIONS_TOTAL.labels(model=result.model_used, label_name=result.label_name).inc()
    PREDICTION_CONFIDENCE.labels(model=result.model_used).observe(result.confidence)

    return PredictResponse(
        label_id=result.label_id,
        label_name=result.label_name,
        model_used=result.model_used,
        confidence=result.confidence,
        class_scores=result.class_scores,
    )
