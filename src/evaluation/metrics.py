from PIL.IcnsImagePlugin import fp
import numpy as np

from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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
    
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    if (tn + fp) > 0:
        metrics["specificity"] = tn / (tn + fp)
    else:
        metrics["specificity"] = 0.0

    return metrics