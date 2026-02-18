from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

REQUESTS_TOTAL = Counter(
    "news_api_requests_total",
    "Total API requests.",
    labelnames=("path", "method", "status"),
)

PREDICTIONS_TOTAL = Counter(
    "news_api_predictions_total",
    "Total prediction requests by model and predicted label.",
    labelnames=("model", "label_name"),
)

ERRORS_TOTAL = Counter(
    "news_api_errors_total",
    "Total handled API errors.",
    labelnames=("path", "error_type"),
)

REQUEST_LATENCY_SECONDS = Histogram(
    "news_api_request_latency_seconds",
    "Request latency in seconds.",
    labelnames=("path", "method"),
    buckets=(0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.5, 1.0, 2.0, 5.0),
)

PREDICTION_CONFIDENCE = Histogram(
    "news_api_prediction_confidence",
    "Confidence distribution for predictions.",
    labelnames=("model",),
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

INCIDENT_SUMMARIES_TOTAL = Counter(
    "news_api_incident_summaries_total",
    "Count of incident summaries by status and model.",
    labelnames=("status", "model"),
)

INCIDENT_SUMMARY_LATENCY_SECONDS = Histogram(
    "news_api_incident_summary_latency_seconds",
    "Latency for generating incident summaries.",
    labelnames=("model",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0),
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
