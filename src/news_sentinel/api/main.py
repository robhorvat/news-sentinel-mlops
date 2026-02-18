from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from news_sentinel.inference.predictors import PredictorManager
from news_sentinel.inference.schemas import (
    HealthResponse,
    IncidentSummaryRequest,
    IncidentSummaryResponse,
    PredictRequest,
    PredictResponse,
)
from news_sentinel.llm.gemini_summary import (
    GeminiIncidentSummarizer,
    GeminiSummaryRuntimeError,
    GeminiSummaryUnavailableError,
    build_local_incident_summary,
)
from news_sentinel.observability.metrics import (
    ERRORS_TOTAL,
    INCIDENT_SUMMARIES_TOTAL,
    INCIDENT_SUMMARY_LATENCY_SECONDS,
    PREDICTION_CONFIDENCE,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
    metrics_response,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor_manager = PredictorManager.from_artifacts()
    app.state.incident_summarizer = None
    app.state.incident_summary_reason = "Gemini summary is disabled."

    try:
        summarizer = GeminiIncidentSummarizer.from_env()
        app.state.incident_summarizer = summarizer
        app.state.incident_summary_reason = f"enabled ({summarizer.model_name})"
    except GeminiSummaryUnavailableError as exc:
        app.state.incident_summary_reason = str(exc)

    yield


app = FastAPI(title="News Sentinel API", version="0.1.0", lifespan=lifespan)
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")


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
    incident_reason = getattr(app.state, "incident_summary_reason", "Gemini summary status unknown.")
    return {
        "service": "news-sentinel-api",
        "status": "ok",
        "dashboard": "/dashboard",
        "docs": "/docs",
        "health": "/healthz",
        "predict": "/predict",
        "incident_summary": "/incident-summary",
        "incident_summary_status": incident_reason,
        "metrics": "/metrics",
    }


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


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


@app.post("/incident-summary", response_model=IncidentSummaryResponse)
def incident_summary(payload: IncidentSummaryRequest) -> IncidentSummaryResponse:
    summarizer = getattr(app.state, "incident_summarizer", None)
    reason = getattr(app.state, "incident_summary_reason", "Gemini summary unavailable.")
    manager: PredictorManager = app.state.predictor_manager

    if summarizer is None:
        INCIDENT_SUMMARIES_TOTAL.labels(status="unavailable", model="none").inc()
        ERRORS_TOTAL.labels(path="/incident-summary", error_type="SummaryUnavailable").inc()
        raise HTTPException(status_code=503, detail=reason)

    try:
        result = manager.predict(text=payload.text, requested_model=payload.model)
    except ValueError as exc:
        ERRORS_TOTAL.labels(path="/incident-summary", error_type="ValueError").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = perf_counter()
    try:
        summary = summarizer.summarize(
            headline=payload.text,
            predicted_label=result.label_name,
            model_used=result.model_used,
            confidence=result.confidence,
            class_scores=result.class_scores,
        )
    except GeminiSummaryRuntimeError as exc:
        INCIDENT_SUMMARIES_TOTAL.labels(status="fallback", model=result.model_used).inc()
        ERRORS_TOTAL.labels(path="/incident-summary", error_type=type(exc).__name__).inc()
        summary = build_local_incident_summary(
            headline=payload.text,
            predicted_label=result.label_name,
            model_used=result.model_used,
            confidence=result.confidence,
            class_scores=result.class_scores,
            failure_note=str(exc),
        )
    except Exception as exc:
        INCIDENT_SUMMARIES_TOTAL.labels(status="error", model=result.model_used).inc()
        ERRORS_TOTAL.labels(path="/incident-summary", error_type=type(exc).__name__).inc()
        raise HTTPException(status_code=502, detail="Failed to generate incident summary.") from exc
    finally:
        elapsed = perf_counter() - started
        INCIDENT_SUMMARY_LATENCY_SECONDS.labels(model=result.model_used).observe(elapsed)

    INCIDENT_SUMMARIES_TOTAL.labels(status="success", model=result.model_used).inc()
    return IncidentSummaryResponse(
        summary=summary,
        predicted_label=result.label_name,
        model_used=result.model_used,
        confidence=result.confidence,
    )
