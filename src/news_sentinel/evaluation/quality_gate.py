from __future__ import annotations

from typing import Dict


def evaluate_quality_gate(report: Dict, p95_latency_ms_max: float = 120.0) -> Dict[str, object]:
    models = report.get("models", {})
    baseline = models.get("baseline")
    textcnn = models.get("textcnn")

    if not baseline or not textcnn:
        return {
            "status": "pending",
            "reason": "missing required model metrics",
            "checks": {},
            "thresholds": {
                "macro_f1_min_vs_baseline": 0.0,
                "p95_latency_ms_max": p95_latency_ms_max,
            },
        }

    baseline_f1 = float(baseline.get("macro_f1", 0.0))
    textcnn_f1 = float(textcnn.get("macro_f1", 0.0))
    textcnn_p95 = float(textcnn.get("latency", {}).get("p95_ms", 0.0))

    checks = {
        "macro_f1_vs_baseline": {
            "actual": textcnn_f1 - baseline_f1,
            "threshold": 0.0,
            "passed": textcnn_f1 >= baseline_f1,
        },
        "textcnn_p95_latency_ms": {
            "actual": textcnn_p95,
            "threshold": p95_latency_ms_max,
            "passed": textcnn_p95 <= p95_latency_ms_max,
        },
    }

    status = "pass" if all(v["passed"] for v in checks.values()) else "fail"

    return {
        "status": status,
        "reason": "baseline_vs_textcnn_comparison",
        "checks": checks,
        "thresholds": {
            "macro_f1_min_vs_baseline": 0.0,
            "p95_latency_ms_max": p95_latency_ms_max,
        },
    }
