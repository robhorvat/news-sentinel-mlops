from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class BaselineTrainingInput:
    train_texts: List[str]
    train_labels: List[int]


def build_baseline_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(C=1.0, class_weight="balanced", random_state=42),
            ),
        ]
    )


def train_baseline_model(inputs: BaselineTrainingInput) -> Pipeline:
    model = build_baseline_pipeline()
    model.fit(inputs.train_texts, inputs.train_labels)
    return model
