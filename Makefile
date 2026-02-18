PYTHON ?= venv/bin/python
IMAGE_CPU ?= news-sentinel-mlops:cpu
IMAGE_TORCH ?= news-sentinel-mlops:torch

.PHONY: prep-data train-baseline train-textcnn-quick train-textcnn show-model-registry check-drift eval-report quality-gate quality-gate-soft install-torch install-gemini run-api docker-build docker-build-torch docker-run docker-run-torch minikube-namespace minikube-build-image minikube-secret-from-env minikube-deploy minikube-status ci test lint

prep-data:
	$(PYTHON) scripts/prepare_ag_news.py --output-dir data/ag_news/processed

train-baseline:
	$(PYTHON) scripts/train_baseline.py \
		--train-file data/ag_news/processed/train.jsonl \
		--test-file data/ag_news/processed/test.jsonl \
		--model-out artifacts/baseline_tfidf_svc.joblib \
		--report-out reports/baseline_eval.json

install-torch:
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install -r requirements-torch.txt

install-gemini:
	$(PYTHON) -m pip install -r requirements-llm.txt

train-textcnn-quick:
	$(PYTHON) scripts/train_textcnn.py \
		--train-file data/ag_news/processed/train.jsonl \
		--test-file data/ag_news/processed/test.jsonl \
		--output-dir artifacts/textcnn_quick \
		--epochs 2 \
		--batch-size 128 \
		--max-train-samples 20000 \
		--max-test-samples 3000

train-textcnn:
	$(PYTHON) scripts/train_textcnn.py \
		--train-file data/ag_news/processed/train.jsonl \
		--test-file data/ag_news/processed/test.jsonl \
		--output-dir artifacts/textcnn_full \
		--epochs 4 \
		--batch-size 128

show-model-registry:
	$(PYTHON) scripts/show_model_registry.py --registry artifacts/model_registry.jsonl --limit 10

check-drift:
	$(PYTHON) scripts/check_drift.py \
		--reference-file data/ag_news/processed/test.jsonl \
		--current-file data/ag_news/processed/test.jsonl \
		--model-path artifacts/baseline_tfidf_svc.joblib \
		--current-sample-size 3000 \
		--output reports/drift_report.json

eval-report:
	$(PYTHON) scripts/evaluate_stack.py \
		--eval-file data/ag_news/processed/test.jsonl \
		--baseline-model artifacts/baseline_tfidf_svc.joblib \
		--textcnn-checkpoint artifacts/textcnn_quick/textcnn.pt \
		--max-samples 2000 \
		--output-json reports/eval_report.json \
		--output-md reports/eval_report.md

quality-gate-soft: eval-report
	$(PYTHON) scripts/check_quality_gate.py --report reports/eval_report.json --allow-fail --allow-pending

quality-gate: eval-report
	$(PYTHON) scripts/check_quality_gate.py --report reports/eval_report.json

run-api:
	PYTHONPATH=src $(PYTHON) -m uvicorn news_sentinel.api.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t $(IMAGE_CPU) .

docker-build-torch:
	docker build --build-arg INSTALL_TORCH=1 -t $(IMAGE_TORCH) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE_CPU)

docker-run-torch:
	docker run --rm -p 8001:8000 $(IMAGE_TORCH)

minikube-namespace:
	kubectl create namespace news-sentinel --dry-run=client -o yaml | kubectl apply -f -

minikube-build-image:
	minikube image build -t $(IMAGE_CPU) .

minikube-secret-from-env: minikube-namespace
	@if [ -f .env ]; then \
		set -a; . ./.env; set +a; \
		kubectl -n news-sentinel create secret generic news-sentinel-secrets \
			--from-literal=GEMINI_API_KEY="$${GEMINI_API_KEY:-}" \
			--dry-run=client -o yaml | kubectl apply -f -; \
	else \
		echo \"No .env file found, skipping secret creation\"; \
	fi

minikube-deploy: minikube-namespace
	kubectl -n news-sentinel apply -f k8s/configmap.yaml
	kubectl -n news-sentinel apply -f k8s/deployment.yaml
	kubectl -n news-sentinel apply -f k8s/service.yaml

minikube-status:
	kubectl -n news-sentinel get pods,svc

ci: lint test
	$(PYTHON) scripts/prepare_ag_news.py --output-dir data/ag_news/processed
	$(PYTHON) scripts/train_baseline.py \
		--train-file data/ag_news/processed/train.jsonl \
		--test-file data/ag_news/processed/test.jsonl \
		--model-out artifacts/baseline_tfidf_svc.joblib \
		--report-out reports/baseline_eval.json
	$(PYTHON) scripts/evaluate_stack.py \
		--eval-file data/ag_news/processed/test.jsonl \
		--baseline-model artifacts/baseline_tfidf_svc.joblib \
		--textcnn-checkpoint artifacts/textcnn_quick/textcnn.pt \
		--max-samples 500 \
		--output-json reports/eval_report.json \
		--output-md reports/eval_report.md
	$(PYTHON) scripts/check_quality_gate.py --report reports/eval_report.json --allow-fail --allow-pending

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src scripts
