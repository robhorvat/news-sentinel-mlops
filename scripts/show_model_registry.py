from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from news_sentinel.model_registry import read_registry_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show model registry entries.")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/model_registry.jsonl"))
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = read_registry_entries(args.registry)
    print(f"Registry: {args.registry}")
    print(f"Total entries: {len(entries)}")

    recent = entries[-args.limit :]
    if not recent:
        return

    print(f"Showing last {len(recent)}:")
    for entry in recent:
        model = entry.get("model_name")
        run_id = entry.get("run_id")
        macro_f1 = entry.get("metrics", {}).get("macro_f1")
        accuracy = entry.get("metrics", {}).get("accuracy")
        print(f"- model={model} run_id={run_id} macro_f1={macro_f1} accuracy={accuracy}")


if __name__ == "__main__":
    main()
