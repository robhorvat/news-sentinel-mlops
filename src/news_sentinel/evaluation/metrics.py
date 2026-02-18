from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def classification_report_dict(y_true: List[int], y_pred: List[int]) -> Dict[str, object]:
    per_class_recall_values = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    per_class_recall = {
        str(label): float(score) for label, score in zip([0, 1, 2, 3], per_class_recall_values)
    }

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(len(y_true)),
    }
