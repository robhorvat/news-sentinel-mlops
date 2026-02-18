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


def test_metrics_endpoint_exposes_custom_metrics() -> None:
    app.state.predictor_manager = _FakeManager()
    client = TestClient(app)

    response = client.post("/predict", json={"text": "Team wins league title", "model": "auto"})
    assert response.status_code == 200

    metrics = client.get("/metrics")
    body = metrics.text

    assert "news_api_requests_total" in body
    assert "news_api_request_latency_seconds" in body
    assert "news_api_predictions_total" in body
    assert "news_api_prediction_confidence" in body
