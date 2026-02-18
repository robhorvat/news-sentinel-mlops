from fastapi.testclient import TestClient

from news_sentinel.api.main import app
from news_sentinel.llm.gemini_summary import GeminiSummaryRuntimeError


class _FakeResult:
    label_id = 2
    label_name = "Business"
    model_used = "baseline"
    confidence = 0.91
    class_scores = {"0": 0.01, "1": 0.02, "2": 0.91, "3": 0.06}


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
        return "- Situation: Classified headline.\n- Evidence: Baseline predicted Business."


class _FailingSummarizer:
    def summarize(
        self,
        headline: str,
        predicted_label: str,
        model_used: str,
        confidence: float,
        class_scores: dict[str, float],
    ) -> str:
        _ = (headline, predicted_label, model_used, confidence, class_scores)
        raise GeminiSummaryRuntimeError("Gemini API error (401): invalid key")


def test_healthz_endpoint() -> None:
    app.state.predictor_manager = _FakeManager()
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint_contains_dashboard_link() -> None:
    app.state.predictor_manager = _FakeManager()
    app.state.incident_summary_reason = "Gemini summary disabled."
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["dashboard"] == "/dashboard"
    assert data["incident_summary"] == "/incident-summary"


def test_dashboard_endpoint_serves_html() -> None:
    app.state.predictor_manager = _FakeManager()
    client = TestClient(app)

    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "News Sentinel Control Room" in response.text


def test_predict_endpoint() -> None:
    app.state.predictor_manager = _FakeManager()
    client = TestClient(app)

    response = client.post("/predict", json={"text": "Stocks rally on earnings", "model": "auto"})
    assert response.status_code == 200
    data = response.json()
    assert data["label_name"] == "Business"
    assert data["model_used"] == "baseline"


def test_incident_summary_unavailable() -> None:
    app.state.predictor_manager = _FakeManager()
    app.state.incident_summarizer = None
    app.state.incident_summary_reason = "Gemini summary disabled for test."
    client = TestClient(app)

    response = client.post("/incident-summary", json={"text": "Stocks rally on earnings", "model": "auto"})
    assert response.status_code == 503
    assert "Gemini summary disabled for test." in response.json()["detail"]


def test_incident_summary_success() -> None:
    app.state.predictor_manager = _FakeManager()
    app.state.incident_summarizer = _FakeSummarizer()
    app.state.incident_summary_reason = "enabled (fake)"
    client = TestClient(app)

    response = client.post("/incident-summary", json={"text": "Stocks rally on earnings", "model": "auto"})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "Business"
    assert data["model_used"] == "baseline"
    assert "Situation" in data["summary"]


def test_incident_summary_runtime_failure_falls_back_to_local_summary() -> None:
    app.state.predictor_manager = _FakeManager()
    app.state.incident_summarizer = _FailingSummarizer()
    app.state.incident_summary_reason = "enabled (fake)"
    client = TestClient(app)

    response = client.post("/incident-summary", json={"text": "Stocks rally on earnings", "model": "auto"})
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "Business"
    assert "Gemini fallback" in data["summary"]
