import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    metrics["f1_score"] = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )

    except:

        metrics["roc_auc"] = np.nan

    return metrics