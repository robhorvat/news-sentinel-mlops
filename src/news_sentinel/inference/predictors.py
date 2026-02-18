from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from news_sentinel.data.ag_news import AG_NEWS_LABELS
from news_sentinel.data.preprocess import clean_text


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


@dataclass
class PredictionResult:
    label_id: int
    label_name: str
    model_used: str
    confidence: float
    class_scores: Dict[str, float]


class SklearnBaselinePredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        scores = self.pipeline.decision_function([cleaned])[0]
        probs = _softmax(np.array(scores, dtype=float))
        pred_id = int(np.argmax(probs))

        class_scores = {str(i): float(probs[i]) for i in range(4)}
        return PredictionResult(
            label_id=pred_id,
            label_name=AG_NEWS_LABELS[pred_id],
            model_used="baseline",
            confidence=float(probs[pred_id]),
            class_scores=class_scores,
        )


class TextCnnPredictor:
    def __init__(self, checkpoint_path: Path):
        import torch

        from news_sentinel.models.textcnn import TextCNN
        from news_sentinel.models.torch_text_data import Vocabulary

        self._torch = torch
        self.checkpoint_path = checkpoint_path

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["config"]
        idx_to_token: List[str] = checkpoint["vocab"]
        token_to_idx = {tok: i for i, tok in enumerate(idx_to_token)}
        self.vocab = Vocabulary(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

        self.max_length = int(config["max_length"])
        self.model = TextCNN(
            vocab_size=len(self.vocab),
            embedding_dim=int(config["embedding_dim"]),
            num_filters=int(config["num_filters"]),
            num_classes=int(config["num_classes"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        cleaned = clean_text(text)
        token_ids = self.vocab.encode(cleaned, max_length=self.max_length)
        x = self._torch.tensor([token_ids], dtype=self._torch.long)

        with self._torch.no_grad():
            logits = self.model(x)
            probs = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        class_scores = {str(i): float(probs[i]) for i in range(4)}
        return PredictionResult(
            label_id=pred_id,
            label_name=AG_NEWS_LABELS[pred_id],
            model_used="textcnn",
            confidence=float(probs[pred_id]),
            class_scores=class_scores,
        )


class PredictorManager:
    def __init__(self, predictors: Dict[str, object]):
        self.predictors = predictors

    @classmethod
    def from_artifacts(
        cls,
        baseline_path: Path | None = None,
        textcnn_path: Path | None = None,
        registry_latest_path: Path | None = None,
    ) -> "PredictorManager":
        predictors: Dict[str, object] = {}

        if baseline_path is None:
            baseline_path = Path(os.getenv("BASELINE_MODEL_PATH", "artifacts/baseline_tfidf_svc.joblib"))
        if textcnn_path is None:
            textcnn_path = Path(os.getenv("TEXTCNN_CHECKPOINT_PATH", "artifacts/textcnn_quick/textcnn.pt"))
        if registry_latest_path is None:
            registry_latest_path = Path(
                os.getenv("MODEL_REGISTRY_LATEST_PATH", "artifacts/model_registry_latest.json")
            )

        if baseline_path.exists():
            predictors["baseline"] = SklearnBaselinePredictor(baseline_path)

        if textcnn_path.exists():
            try:
                predictors["textcnn"] = TextCnnPredictor(textcnn_path)
            except Exception:
                pass

        # Optional: if registry exists, this can influence default order in the future.
        if registry_latest_path.exists():
            _ = json.loads(registry_latest_path.read_text(encoding="utf-8"))

        return cls(predictors=predictors)

    def available_models(self) -> List[str]:
        return list(self.predictors.keys())

    def _resolve_model(self, requested: str) -> str:
        if requested != "auto":
            if requested not in self.predictors:
                raise ValueError(f"Requested model '{requested}' is unavailable.")
            return requested

        if "baseline" in self.predictors:
            return "baseline"
        if "textcnn" in self.predictors:
            return "textcnn"
        raise ValueError("No model artifacts are available.")

    def predict(self, text: str, requested_model: str) -> PredictionResult:
        model_name = self._resolve_model(requested_model)
        predictor = self.predictors[model_name]
        return predictor.predict(text)
