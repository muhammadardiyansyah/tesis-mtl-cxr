"""Mask-aware evaluation metrics for multi-task binary classification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_TASK_NAMES = ("cardiomegaly", "tuberculosis")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | int]:
    """Calculate robust metrics for one binary task."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("y_true, y_pred, dan y_prob harus memiliki panjang sama.")
    if len(y_true) == 0:
        return {
            "n": 0,
            "positive": 0,
            "negative": 0,
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "f1_score": float("nan"),
            "roc_auc": float("nan"),
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auc = (
        float(roc_auc_score(y_true, y_prob))
        if np.unique(y_true).size == 2
        else float("nan")
    )

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int(len(y_true) - y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": f1,
        "f1_score": f1,
        "roc_auc": auc,
    }


def calculate_multitask_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    masks: np.ndarray,
    task_names: Sequence[str] = DEFAULT_TASK_NAMES,
    thresholds: float | Sequence[float] = 0.5,
) -> dict[str, dict[str, float | int]]:
    """Calculate each task only on rows whose corresponding mask is active."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    masks = np.asarray(masks)

    if labels.shape != probabilities.shape or labels.shape != masks.shape:
        raise ValueError("labels, probabilities, dan masks harus berbentuk sama.")
    if labels.ndim != 2 or labels.shape[1] != len(task_names):
        raise ValueError("Dimensi kedua input harus sama dengan jumlah task.")

    if np.isscalar(thresholds):
        threshold_values = [float(thresholds)] * len(task_names)
    else:
        threshold_values = [float(value) for value in thresholds]
        if len(threshold_values) != len(task_names):
            raise ValueError("Jumlah threshold harus sama dengan jumlah task.")

    results: dict[str, dict[str, float | int]] = {}
    for task_index, task_name in enumerate(task_names):
        active = masks[:, task_index] > 0
        task_labels = labels[active, task_index].astype(np.int64)
        task_probs = probabilities[active, task_index].astype(np.float64)
        task_preds = (task_probs >= threshold_values[task_index]).astype(np.int64)
        results[str(task_name)] = calculate_metrics(
            task_labels,
            task_preds,
            task_probs,
        )
    return results
