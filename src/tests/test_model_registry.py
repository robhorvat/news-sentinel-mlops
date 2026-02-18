from pathlib import Path

from news_sentinel.model_registry import (
    append_registry_entry,
    read_registry_entries,
    write_latest_snapshot,
)


def test_registry_append_and_read(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    entry = {"run_id": "run-1", "metrics": {"macro_f1": 0.8}}

    append_registry_entry(registry, entry)
    loaded = read_registry_entries(registry)

    assert len(loaded) == 1
    assert loaded[0]["run_id"] == "run-1"


def test_latest_snapshot_written(tmp_path: Path) -> None:
    latest = tmp_path / "latest.json"
    entry = {"run_id": "run-2"}

    write_latest_snapshot(latest, entry)

    assert latest.exists()
