from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class GitSnapshot:
    commit: str
    branch: str
    dirty: bool


def _run_git(args: List[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL
    ).strip()


def get_git_snapshot(repo_root: Path) -> GitSnapshot:
    try:
        commit = _run_git(["rev-parse", "--short", "HEAD"], repo_root)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        status = _run_git(["status", "--porcelain"], repo_root)
        dirty = bool(status.strip())
        return GitSnapshot(commit=commit, branch=branch, dirty=dirty)
    except Exception:
        return GitSnapshot(commit="unknown", branch="unknown", dirty=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_id(prefix: str, git_commit: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S%z")
    return f"{prefix}-{ts}-{git_commit}"


def append_registry_entry(registry_path: Path, entry: Dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def write_latest_snapshot(latest_path: Path, entry: Dict[str, Any]) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)


def read_registry_entries(registry_path: Path) -> List[Dict[str, Any]]:
    if not registry_path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with registry_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
