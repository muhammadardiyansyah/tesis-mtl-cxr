import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)


def plot_roc_curve(
    y_true,
    y_prob,
    class_name="Class"
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"ROC Curve - {class_name}"
    )

    plt.legend()

    plt.show()