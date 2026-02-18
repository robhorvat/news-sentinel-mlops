from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DriftThresholds:
    class_prior_tvd_warn: float = 0.15
    embedding_shift_warn: float = 0.20


def label_distribution(labels: Iterable[int], n_classes: int = 4) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=float)
    for label in labels:
        counts[int(label)] += 1.0
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def class_prior_total_variation(ref_labels: List[int], cur_labels: List[int]) -> Dict[str, object]:
    ref_dist = label_distribution(ref_labels)
    cur_dist = label_distribution(cur_labels)
    tvd = 0.5 * float(np.abs(ref_dist - cur_dist).sum())

    return {
        "reference_distribution": {str(i): float(ref_dist[i]) for i in range(len(ref_dist))},
        "current_distribution": {str(i): float(cur_dist[i]) for i in range(len(cur_dist))},
        "total_variation_distance": tvd,
    }


def tfidf_centroid_cosine_distance(
    reference_texts: List[str], current_texts: List[str], vectorizer
) -> Dict[str, float]:
    ref_matrix = vectorizer.transform(reference_texts)
    cur_matrix = vectorizer.transform(current_texts)

    ref_centroid = np.asarray(ref_matrix.mean(axis=0)).reshape(1, -1)
    cur_centroid = np.asarray(cur_matrix.mean(axis=0)).reshape(1, -1)

    similarity = float(cosine_similarity(ref_centroid, cur_centroid)[0, 0])
    distance = 1.0 - similarity

    return {
        "cosine_similarity": similarity,
        "cosine_distance": distance,
    }


def summarize_drift(
    class_prior_tvd: float,
    embedding_distance: float,
    thresholds: DriftThresholds,
) -> Dict[str, object]:
    class_prior_flag = class_prior_tvd >= thresholds.class_prior_tvd_warn
    embedding_flag = embedding_distance >= thresholds.embedding_shift_warn

    status = "warn" if (class_prior_flag or embedding_flag) else "pass"
    return {
        "status": status,
        "checks": {
            "class_prior_tvd_warn": {
                "actual": class_prior_tvd,
                "threshold": thresholds.class_prior_tvd_warn,
                "flagged": class_prior_flag,
            },
            "embedding_shift_warn": {
                "actual": embedding_distance,
                "threshold": thresholds.embedding_shift_warn,
                "flagged": embedding_flag,
            },
        },
    }
