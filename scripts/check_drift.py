from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from news_sentinel.data.ag_news import read_jsonl
from news_sentinel.drift.checks import (
    DriftThresholds,
    class_prior_total_variation,
    summarize_drift,
    tfidf_centroid_cosine_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data drift checks for AG News pipeline.")
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--current-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/baseline_tfidf_svc.joblib"))
    parser.add_argument("--current-sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.json"))
    return parser.parse_args()


def _extract(rows: list[dict]) -> tuple[list[str], list[int]]:
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def main() -> None:
    args = parse_args()

    reference_rows = read_jsonl(args.reference_file)
    current_rows = read_jsonl(args.current_file)

    if args.current_sample_size > 0 and args.current_sample_size < len(current_rows):
        random.Random(args.seed).shuffle(current_rows)
        current_rows = current_rows[: args.current_sample_size]

    ref_texts, ref_labels = _extract(reference_rows)
    cur_texts, cur_labels = _extract(current_rows)

    baseline = joblib.load(args.model_path)
    vectorizer = baseline.named_steps["tfidf"]

    class_prior = class_prior_total_variation(ref_labels, cur_labels)
    embedding = tfidf_centroid_cosine_distance(ref_texts, cur_texts, vectorizer)

    thresholds = DriftThresholds()
    summary = summarize_drift(
        class_prior_tvd=class_prior["total_variation_distance"],
        embedding_distance=embedding["cosine_distance"],
        thresholds=thresholds,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_file": str(args.reference_file),
        "current_file": str(args.current_file),
        "n_reference": len(ref_texts),
        "n_current": len(cur_texts),
        "class_prior": class_prior,
        "embedding_shift": embedding,
        "thresholds": {
            "class_prior_tvd_warn": thresholds.class_prior_tvd_warn,
            "embedding_shift_warn": thresholds.embedding_shift_warn,
        },
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved drift report: {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
