# News Sentinel MLOps

News Sentinel is a production-minded NLP MLOps project for AG News classification.

It demonstrates a complete ML lifecycle with measurable checks:
- classical baseline (TF-IDF + LinearSVC)
- PyTorch neural model (TextCNN)
- model registry metadata
- FastAPI inference service
- Prometheus-compatible metrics
- drift checks
- regression quality gates
- Docker + Minikube deployment
- CI automation

## Architecture Story

AG News dataset
-> preprocessing (clean/tokenize)
-> Baseline model (TF-IDF + LinearSVC)
-> PyTorch model (TextCNN)
-> artifact + metadata registry
-> FastAPI inference (`/predict`)
-> optional Gemini incident digest (`/incident-summary`)
-> observability (`/metrics`)
-> drift checks (class prior + TF-IDF centroid shift)
-> evaluation harness + quality gate
-> Docker + Minikube + CI

## Why each technology is here

- `scikit-learn`: strong classical baseline for ablations.
- `PyTorch`: trainable neural text classifier for model comparison.
- `FastAPI`: typed, testable, low-latency inference API.
- `prometheus-client`: export runtime ML/API health metrics.
- `MLflow`: experiment tracking extension point for upcoming runs.
- `Docker`: reproducible runtime and onboarding.
- `Kubernetes/minikube`: realistic deployment shape with probes.
- `GitHub Actions`: automated regression checks before merge.

## Current Progress (16-step plan)

- Completed: Steps 1-15 (bootstrap through optional Gemini incident summary endpoint).
- In progress: Step 16 README and portfolio polish.

## Reproducible Workflow

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

# data + baseline
make prep-data
make train-baseline

# optional torch path
make install-torch
make train-textcnn-quick

# optional Gemini summary dependency
make install-gemini

# evaluation and gates
make eval-report
make quality-gate-soft

# api
make run-api
# open http://localhost:8000/dashboard

# tests
make test
```

## Evaluation Snapshot

From latest `make eval-report` (2000 AG News test samples):

- Baseline macro-F1: `0.9205`
- TextCNN macro-F1: `0.8311`
- TextCNN p95 latency: `~0.42 ms`
- Gate status: `fail` (TextCNN under baseline, latency passes)

This fail is expected right now and useful: it prevents promoting a worse model.

## API Endpoints

- `GET /dashboard` (visual demo UI)
- `GET /healthz`
- `GET /models`
- `POST /predict`
- `POST /incident-summary` (optional Gemini-backed digest)
- `GET /metrics`

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{"text":"Stocks rally after earnings beat","model":"auto"}'
```

## Docker

```bash
make docker-build
make docker-run
```

Torch-enabled image (optional):

```bash
make docker-build-torch
make docker-run-torch
```

## Minikube

```bash
make minikube-build-image
make minikube-deploy
make minikube-status
minikube service -n news-sentinel news-sentinel-api --url
```

## Optional Gemini Incident Summary

The API exposes `POST /incident-summary` behind a feature flag.

Local run:

```bash
make install-gemini
export GEMINI_SUMMARY_ENABLED=1
export GEMINI_API_KEY="AIza..."   # use a real key, not the literal string "your_key"
export GEMINI_SUMMARY_MODEL="gemini-2.5-flash"
export GEMINI_SUMMARY_FALLBACK_MODELS="gemini-2.0-flash,gemini-1.5-flash"
make run-api
```

Kubernetes run:

- `k8s/configmap.yaml` sets `GEMINI_SUMMARY_ENABLED=0` by default.
- `make minikube-secret-from-env` injects `GEMINI_API_KEY` from `.env`.
- Set `GEMINI_SUMMARY_ENABLED=1` in `k8s/configmap.yaml` (or `kubectl set env`) to activate.

If Gemini returns auth/quota/model errors at runtime, the endpoint now falls back to a deterministic local incident summary so the dashboard remains demo-safe.

## CI

Workflow: `.github/workflows/ci.yml`

Runs on push/PR:
- compile-based lint
- unit tests
- data prep + baseline training
- evaluation + soft gate check
- Docker build

## Limitations and Next Iteration

- TextCNN currently underperforms the classical baseline.
- No external streaming ingestion yet (batch-first by design).
- Gemini digest quality is prompt-based and should be benchmarked before production use.

Next high-value steps:
- stronger PyTorch training recipe (class balancing, scheduler, more epochs)
- model promotion policy based on repeated eval runs
- optional LLM digest endpoint behind feature flag
