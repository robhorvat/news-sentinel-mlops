from pathlib import Path

import yaml


def _load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_deployment_has_health_probes() -> None:
    doc = _load_yaml("k8s/deployment.yaml")
    container = doc["spec"]["template"]["spec"]["containers"][0]

    assert container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"


def test_service_exposes_http_port() -> None:
    doc = _load_yaml("k8s/service.yaml")
    ports = doc["spec"]["ports"]
    assert any(p["port"] == 8000 for p in ports)


def test_configmap_contains_model_paths() -> None:
    doc = _load_yaml("k8s/configmap.yaml")
    data = doc["data"]
    assert "BASELINE_MODEL_PATH" in data
    assert "TEXTCNN_CHECKPOINT_PATH" in data
    assert "GEMINI_SUMMARY_ENABLED" in data
    assert "GEMINI_SUMMARY_MODEL" in data
