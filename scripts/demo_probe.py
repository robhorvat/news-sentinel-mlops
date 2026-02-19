from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a demo probe against the News Sentinel API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--headline",
        default="AI chip startup unveils faster inference hardware",
        help="Headline used for predict and incident-summary calls.",
    )
    parser.add_argument("--output-json", type=Path, default=Path("reports/demo_probe.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/demo_probe.md"))
    parser.add_argument(
        "--require-incident-summary",
        action="store_true",
        help="Fail probe if /incident-summary is not available.",
    )
    return parser.parse_args()


def _request_json(base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    headers = {}
    method = "GET"

    if payload is not None:
        method = "POST"
        headers["content-type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, data=body, headers=headers, method=method)

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw)
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"detail": raw}
        return exc.code, parsed


def _request_text(base_url: str, path: str) -> tuple[int, str]:
    url = f"{base_url.rstrip('/')}{path}"
    req = request.Request(url=url, method="GET")
    with request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read().decode("utf-8")


def _sum_metric(metrics_text: str, name: str) -> float:
    total = 0.0
    for line in metrics_text.splitlines():
        if not line.startswith(name + "{"):
            continue
        try:
            total += float(line.split(" ")[-1])
        except Exception:
            continue
    return total


def _write_markdown(path: Path, report: dict) -> None:
    checks = report.get("checks", {})
    predict = report.get("predict", {})
    incident = report.get("incident_summary", {})
    metrics = report.get("metrics", {})

    lines = [
        "# Demo Probe Report",
        "",
        f"- Timestamp: {report['timestamp_utc']}",
        f"- Base URL: {report['base_url']}",
        f"- Probe status: **{report['status']}**",
        "",
        "## Endpoint Checks",
        "",
        f"- `/`: {checks.get('root', '-')}",
        f"- `/healthz`: {checks.get('healthz', '-')}",
        f"- `/models`: {checks.get('models', '-')}",
        f"- `/predict`: {checks.get('predict', '-')}",
        f"- `/incident-summary`: {checks.get('incident_summary', '-')}",
        f"- `/metrics`: {checks.get('metrics', '-')}",
        "",
        "## Prediction",
        "",
        f"- Label: {predict.get('label_name', '-')}",
        f"- Model: {predict.get('model_used', '-')}",
        f"- Confidence: {predict.get('confidence', 0.0):.4f}",
        "",
        "## Incident Summary",
        "",
        f"- Enabled status from `/`: {report['incident_status']}",
        f"- Four-section format: {incident.get('has_all_sections')}",
        f"- Contains fallback marker: {incident.get('contains_fallback_marker')}",
        "",
        "## Metrics Snapshot",
        "",
        f"- `news_api_requests_total` sum: {metrics.get('requests_total', 0.0):.0f}",
        f"- `news_api_predictions_total` sum: {metrics.get('predictions_total', 0.0):.0f}",
        f"- `news_api_errors_total` sum: {metrics.get('errors_total', 0.0):.0f}",
    ]

    if report.get("errors"):
        lines.extend(["", "## Errors"])
        for err in report["errors"]:
            lines.append(f"- {err}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    errors_found: list[str] = []

    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "headline": args.headline,
        "checks": {},
        "errors": errors_found,
    }

    try:
        root_code, root = _request_json(args.base_url, "/")
    except Exception as exc:
        report["checks"]["root"] = 0
        report["incident_status"] = "unreachable"
        errors_found.append(f"Could not connect to {args.base_url}: {exc}")
        report["status"] = "fail"
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_markdown(args.output_md, report)
        print(f"Saved demo probe JSON: {args.output_json}")
        print(f"Saved demo probe Markdown: {args.output_md}")
        print("Demo probe status: fail")
        for err in errors_found:
            print(f"- {err}")
        raise SystemExit(1)

    report["checks"]["root"] = root_code
    report["incident_status"] = root.get("incident_summary_status", "unknown")
    if root_code != 200:
        errors_found.append(f"GET / returned {root_code}")

    health_code, health = _request_json(args.base_url, "/healthz")
    report["checks"]["healthz"] = health_code
    if health_code != 200:
        errors_found.append(f"GET /healthz returned {health_code}")
    report["health"] = health

    models_code, models = _request_json(args.base_url, "/models")
    report["checks"]["models"] = models_code
    if models_code != 200:
        errors_found.append(f"GET /models returned {models_code}")
    report["models"] = models

    predict_payload = {"text": args.headline, "model": "auto"}
    predict_code, predict = _request_json(args.base_url, "/predict", payload=predict_payload)
    report["checks"]["predict"] = predict_code
    report["predict"] = predict
    if predict_code != 200:
        errors_found.append(f"POST /predict returned {predict_code}: {predict.get('detail')}")

    incident_code, incident = _request_json(
        args.base_url,
        "/incident-summary",
        payload=predict_payload,
    )
    report["checks"]["incident_summary"] = incident_code
    if incident_code not in (200, 503):
        errors_found.append(f"POST /incident-summary returned {incident_code}: {incident.get('detail')}")
    if args.require_incident_summary and incident_code != 200:
        errors_found.append("Incident summary is required but endpoint is unavailable.")

    summary_text = incident.get("summary", "") if isinstance(incident, dict) else ""
    incident_info = {
        "status_code": incident_code,
        "has_all_sections": all(
            section in summary_text for section in ("Situation", "Evidence", "Risk", "Next Action")
        ),
        "contains_fallback_marker": "Gemini fallback" in summary_text,
        "response": incident,
    }
    report["incident_summary"] = incident_info

    try:
        metrics_code, metrics_text = _request_text(args.base_url, "/metrics")
    except Exception as exc:
        metrics_code, metrics_text = 0, ""
        errors_found.append(f"GET /metrics failed: {exc}")
    report["checks"]["metrics"] = metrics_code
    if metrics_code != 200:
        errors_found.append(f"GET /metrics returned {metrics_code}")
    report["metrics"] = {
        "requests_total": _sum_metric(metrics_text, "news_api_requests_total"),
        "predictions_total": _sum_metric(metrics_text, "news_api_predictions_total"),
        "errors_total": _sum_metric(metrics_text, "news_api_errors_total"),
    }

    report["status"] = "pass" if not errors_found else "fail"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, report)

    print(f"Saved demo probe JSON: {args.output_json}")
    print(f"Saved demo probe Markdown: {args.output_md}")
    print(f"Demo probe status: {report['status']}")

    if errors_found:
        for err in errors_found:
            print(f"- {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
