from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys

import joblib

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from news_sentinel.data.ag_news import read_jsonl
from news_sentinel.evaluation.metrics import classification_report_dict
from news_sentinel.model_registry import (
    append_registry_entry,
    build_run_id,
    get_git_snapshot,
    sha256_file,
    write_latest_snapshot,
)
from news_sentinel.models.baseline import BaselineTrainingInput, train_baseline_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + LinearSVC AG News baseline.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/baseline_tfidf_svc.joblib"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/baseline_eval.json"))
    return parser.parse_args()


def _extract(rows: List[dict]) -> tuple[List[str], List[int]]:
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def main() -> None:
    args = parse_args()

    train_rows = read_jsonl(args.train_file)
    test_rows = read_jsonl(args.test_file)

    train_texts, train_labels = _extract(train_rows)
    test_texts, test_labels = _extract(test_rows)

    model = train_baseline_model(BaselineTrainingInput(train_texts=train_texts, train_labels=train_labels))

    predictions = model.predict(test_texts)
    report = classification_report_dict(test_labels, predictions.tolist())
    report["model_type"] = "tfidf_linear_svc"
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_out)
    with args.report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    repo_root = Path(__file__).resolve().parents[1]
    git = get_git_snapshot(repo_root)
    run_id = build_run_id("baseline_tfidf_svc", git.commit)

    entry = {
        "run_id": run_id,
        "timestamp_utc": report["timestamp_utc"],
        "model_name": "baseline_tfidf_svc",
        "artifact_path": str(args.model_out),
        "artifact_sha256": sha256_file(args.model_out),
        "report_path": str(args.report_out),
        "metrics": {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
        },
        "dataset": {
            "train_file": str(args.train_file),
            "test_file": str(args.test_file),
            "n_train": len(train_texts),
            "n_test": len(test_texts),
        },
        "git": {
            "commit": git.commit,
            "branch": git.branch,
            "dirty": git.dirty,
        },
    }

    registry_path = Path("artifacts/model_registry.jsonl")
    latest_snapshot_path = Path("artifacts/model_registry_latest.json")
    append_registry_entry(registry_path, entry)
    write_latest_snapshot(latest_snapshot_path, entry)

    print(f"Saved model: {args.model_out}")
    print(f"Saved report: {args.report_out}")
    print(f"Updated registry: {registry_path}")
    print(f"Updated latest snapshot: {latest_snapshot_path}")
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
