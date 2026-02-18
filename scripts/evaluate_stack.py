from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from news_sentinel.data.ag_news import read_jsonl
from news_sentinel.evaluation.metrics import classification_report_dict
from news_sentinel.evaluation.quality_gate import evaluate_quality_gate
from news_sentinel.inference.predictors import SklearnBaselinePredictor, TextCnnPredictor



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and TextCNN with gate checks.")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, default=Path("artifacts/baseline_tfidf_svc.joblib"))
    parser.add_argument("--textcnn-checkpoint", type=Path, default=Path("artifacts/textcnn_quick/textcnn.pt"))
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--output-json", type=Path, default=Path("reports/eval_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/eval_report.md"))
    parser.add_argument("--p95-latency-ms-max", type=float, default=120.0)
    return parser.parse_args()


def _latency_summary(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "n": 0}

    sorted_vals = sorted(latencies_ms)
    p95_idx = min(int(0.95 * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return {
        "mean_ms": float(statistics.mean(sorted_vals)),
        "p95_ms": float(sorted_vals[p95_idx]),
        "n": len(sorted_vals),
    }


def _evaluate_model(predictor, texts: list[str], labels: list[int]) -> dict:
    preds = []
    latencies = []

    for text in texts:
        t0 = time.perf_counter()
        pred = predictor.predict(text)
        elapsed = (time.perf_counter() - t0) * 1000.0
        preds.append(pred.label_id)
        latencies.append(elapsed)

    metrics = classification_report_dict(labels, preds)
    metrics["latency"] = _latency_summary(latencies)
    return metrics


def _write_markdown(path: Path, report: dict) -> None:
    gate = report["gate"]
    models = report.get("models", {})

    lines = [
        "# Evaluation Report",
        "",
        f"- Timestamp: {report['timestamp_utc']}",
        f"- Samples: {report['n_samples']}",
        f"- Gate status: **{gate['status']}**",
        "",
        "## Model Metrics",
        "",
        "| Model | Accuracy | Macro-F1 | p95 latency (ms) |",
        "|---|---:|---:|---:|",
    ]

    for model_name in ("baseline", "textcnn"):
        m = models.get(model_name)
        if not m:
            continue
        lines.append(
            f"| {model_name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['latency']['p95_ms']:.2f} |"
        )

    lines.extend([
        "",
        "## Gate Checks",
        "",
        f"- Reason: {gate.get('reason')}",
    ])

    for name, check in gate.get("checks", {}).items():
        lines.append(
            f"- {name}: actual={check['actual']:.4f}, threshold={check['threshold']:.4f}, passed={check['passed']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    rows = read_jsonl(args.eval_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    texts = [r["text"] for r in rows]
    labels = [int(r["label"]) for r in rows]

    model_reports = {}

    if args.baseline_model.exists():
        baseline_predictor = SklearnBaselinePredictor(args.baseline_model)
        model_reports["baseline"] = _evaluate_model(baseline_predictor, texts, labels)

    if args.textcnn_checkpoint.exists():
        textcnn_predictor = TextCnnPredictor(args.textcnn_checkpoint)
        model_reports["textcnn"] = _evaluate_model(textcnn_predictor, texts, labels)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(rows),
        "models": model_reports,
    }

    gate = evaluate_quality_gate(report, p95_latency_ms_max=args.p95_latency_ms_max)
    report["gate"] = gate

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _write_markdown(args.output_md, report)

    print(f"Saved eval report JSON: {args.output_json}")
    print(f"Saved eval report Markdown: {args.output_md}")
    print(f"Quality gate status: {gate['status']}")


if __name__ == "__main__":
    main()
