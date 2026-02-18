from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from news_sentinel.data.ag_news import load_ag_news_splits, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess AG News dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ag_news/processed"),
        help="Directory where train/test jsonl files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_ag_news_splits()

    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"

    write_jsonl(train_path, splits["train"])
    write_jsonl(test_path, splits["test"])

    summary = {
        "train_samples": len(splits["train"]),
        "test_samples": len(splits["test"]),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
