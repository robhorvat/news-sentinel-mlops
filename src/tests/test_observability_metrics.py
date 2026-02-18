from fastapi.testclient import TestClient

from news_sentinel.api.main import app


class _FakeResult:
    label_id = 1
    label_name = "Sports"
    model_used = "baseline"
    confidence = 0.88
    class_scores = {"0": 0.02, "1": 0.88, "2": 0.05, "3": 0.05}


class _FakeManager:
    def available_models(self):
        return ["baseline"]

    def predict(self, text: str, requested_model: str):
        return _FakeResult()


class _FakeSummarizer:
    def summarize(
        self,
        headline: str,
        predicted_label: str,
        model_used: str,
        confidence: float,
        class_scores: dict[str, float],
    ) -> str:
        _ = (headline, predicted_label, model_used, confidence, class_scores)
        return "- Situation: test\n- Evidence: test\n- Risk: low\n- Next Action: monitor"


def test_metrics_endpoint_exposes_custom_metrics() -> None:
    app.state.predictor_manager = _FakeManager()
    app.state.incident_summarizer = _FakeSummarizer()
    app.state.incident_summary_reason = "enabled (fake)"
    client = TestClient(app)

    response = client.post("/predict", json={"text": "Team wins league title", "model": "auto"})
    assert response.status_code == 200
    summary_response = client.post(
        "/incident-summary",
        json={"text": "Team wins league title", "model": "auto"},
    )
    assert summary_response.status_code == 200

    metrics = client.get("/metrics")
    body = metrics.text

    assert "news_api_requests_total" in body
    assert "news_api_request_latency_seconds" in body
    assert "news_api_predictions_total" in body
    assert "news_api_prediction_confidence" in body
    assert "news_api_incident_summaries_total" in body
    assert "news_api_incident_summary_latency_seconds" in body
