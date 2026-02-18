from fastapi.testclient import TestClient

from news_sentinel.api.main import app


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



def test_healthz_endpoint() -> None:
    app.state.predictor_manager = _FakeManager()
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"



def test_predict_endpoint() -> None:
    app.state.predictor_manager = _FakeManager()
    client = TestClient(app)

    response = client.post("/predict", json={"text": "Stocks rally on earnings", "model": "auto"})
    assert response.status_code == 200
    data = response.json()
    assert data["label_name"] == "Business"
    assert data["model_used"] == "baseline"
