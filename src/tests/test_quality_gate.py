from news_sentinel.evaluation.quality_gate import evaluate_quality_gate


def test_quality_gate_pending_when_missing_models() -> None:
    gate = evaluate_quality_gate({"models": {}})
    assert gate["status"] == "pending"


def test_quality_gate_fails_when_textcnn_below_baseline() -> None:
    report = {
        "models": {
            "baseline": {"macro_f1": 0.9, "latency": {"p95_ms": 10}},
            "textcnn": {"macro_f1": 0.8, "latency": {"p95_ms": 50}},
        }
    }
    gate = evaluate_quality_gate(report)
    assert gate["status"] == "fail"


def test_quality_gate_passes_when_thresholds_met() -> None:
    report = {
        "models": {
            "baseline": {"macro_f1": 0.7, "latency": {"p95_ms": 10}},
            "textcnn": {"macro_f1": 0.72, "latency": {"p95_ms": 100}},
        }
    }
    gate = evaluate_quality_gate(report)
    assert gate["status"] == "pass"
