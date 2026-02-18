from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce quality gate from eval report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--allow-pending", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    gate = report.get("gate", {})
    status = gate.get("status", "pending")

    print(f"Quality gate status: {status}")
    print(f"Reason: {gate.get('reason', 'unknown')}")

    thresholds = gate.get("thresholds", {})
    if thresholds:
        print("Thresholds:")
        for key, value in thresholds.items():
            print(f"  - {key}: {value}")

    checks = gate.get("checks", {})
    if checks:
        print("Checks:")
        for key, check in checks.items():
            print(
                f"  - {key}: actual={check['actual']} threshold={check['threshold']} passed={check['passed']}"
            )

    if status == "pass":
        print("Quality gate decision: PASS")
        return

    if status == "fail" and args.allow_fail:
        print("Quality gate decision: PASS (allow-fail)")
        return

    if status == "pending" and args.allow_pending:
        print("Quality gate decision: PASS (allow-pending)")
        return

    print("Quality gate decision: BLOCK")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
