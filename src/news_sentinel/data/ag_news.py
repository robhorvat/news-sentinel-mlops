from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset

from news_sentinel.data.preprocess import clean_text

AG_NEWS_LABELS: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_ag_news_splits() -> Dict[str, List[dict]]:
    dataset = load_dataset("ag_news")
    output: Dict[str, List[dict]] = {}
    for split_name in ("train", "test"):
        rows: List[dict] = []
        for record in dataset[split_name]:
            rows.append(
                {
                    "text": clean_text(record["text"]),
                    "label": int(record["label"]),
                    "label_name": AG_NEWS_LABELS[int(record["label"])],
                }
            )
        output[split_name] = rows
    return output


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
